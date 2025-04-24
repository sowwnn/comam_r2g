import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import Mamba
from models.cnext_mam_3 import create_convnext_mamba_coca, DyT
from models.cross_region_attention import CrossRegionAttention, cross_region_loss, detect_abnormal_regions


class MedicalAttentionGate(nn.Module):
    """
    Cơ chế chú ý đặc biệt cho các đặc trưng y tế quan trọng
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch, seq_len, dim]
        x_t = x.transpose(1, 2)  # [batch, dim, seq_len]
        att = self.pool(x_t).squeeze(-1)  # [batch, dim]
        att = self.fc1(att)
        att = self.act(att)
        att = self.fc2(att)
        att = self.sigmoid(att)
        
        return x * att.unsqueeze(1)  # áp dụng attention theo chiều feature


class MambaDecoder(nn.Module):
    """
    Mamba-based decoder kết hợp với cross-attention
    """
    def __init__(
        self,
        dim,
        num_tokens,
        depth,
        context_dim=None,
        d_state=16,
        d_conv=4,
        expand=2,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, dim) * 0.02)
        
        # Decoder layers với Mamba và cross-attention
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Mamba(
                    d_model=dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                ),
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True),
                nn.Dropout(dropout)
            ]))
        
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)
        
    def forward(self, x, context, mask=None):
        """
        x: [batch, seq_len] - Token indices
        context: [batch, context_len, context_dim] - Image features
        """
        b, n = x.shape
        
        # Token + position embeddings
        x = self.token_emb(x) + self.pos_emb[:, :n]
        
        # Lưu attention weights của cross-attention
        all_cross_attentions = []
        
        # Process through layers
        for norm1, mamba, norm2, cross_attn, dropout in self.layers:
            # Self attention via Mamba
            x_norm = norm1(x)
            mamba_out = mamba(x_norm)
            x = x + dropout(mamba_out)
            
            # Cross attention
            x_norm = norm2(x)
            cross_out, cross_attn_weights = cross_attn(x_norm, context, context, key_padding_mask=mask)
            all_cross_attentions.append(cross_attn_weights)
            x = x + dropout(cross_out)
        
        # Final norm and projection
        x = self.norm(x)
        return self.to_logits(x), all_cross_attentions


class AnatomaMambaCRA(nn.Module):
    """
    AnatomaMamba with Cross-Region Attention: Mô hình kết hợp ConvNeXt-Mamba, 
    Cross-Region Attention và Mamba Decoder để sinh báo cáo y tế từ hình ảnh X-quang
    
    Flow của mô hình:
    1. Image -> ConvNeXt-Mamba + DyT -> Image Features (49 tokens, mỗi token = 1 vùng)
    2. Text -> Token Emb + Mamba -> Text Features
    3. Image Features + Text Features -> Cross-Region Attention -> Enhanced Features
    4. Enhanced Features -> Decoder -> Output
    5. Contrastive Learning giữa Enhanced Features
    """
    def __init__(
        self,
        *,
        dim,
        img_encoder=None,
        image_dim=None,
        num_tokens=10000,
        decoder_depth=6,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        caption_loss_weight=1.0,
        contrastive_loss_weight=1.0,
        cross_region_loss_weight=0.5,
        temperature=0.07,
        max_epochs=50,
        num_heads=8,
        abnormal_terms=None,
        tokenizer=None
    ):
        super().__init__()
        
        # Danh sách từ chỉ bất thường
        self.abnormal_terms = abnormal_terms or [
            'opacity', 'mass', 'nodule', 'abnormal', 'effusion', 'pneumonia',
            'cardiomegaly', 'edema', 'atelectasis', 'consolidation', 'lesion'
        ]
        
        # Lưu trữ tokenizer để sử dụng
        self.tokenizer = tokenizer
        
        # Thêm epoch tracker cho dynamic loss
        self.epoch_tracker = 0
        self.max_epochs = max_epochs
        
        # Loss weights
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.cross_region_loss_weight = cross_region_loss_weight
        self.temperature = temperature
        
        # Image encoder
        assert exists(img_encoder) or exists(image_dim), 'Cần có img_encoder hoặc image_dim'
        
        if exists(img_encoder):
            # Sử dụng pre-trained image encoder
            assert image_dim is None, 'Không thể sử dụng image_dim khi truyền img_encoder'
            self.img_encoder = img_encoder
            image_dim = self.img_encoder.pyramid_decoder.output_conv[0].out_channels
        else:
            image_dim = image_dim
            self.img_encoder = None
        
        # Project image features to embedding dim
        self.img_proj = nn.Sequential(
            DyT(image_dim),
            nn.Linear(image_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Thêm medical attention cho image features
        self.img_med_attention = MedicalAttentionGate(dim)
        
        # Text encoder đơn giản cho contrastive learning
        self.token_emb = nn.Embedding(num_tokens, dim)
        
        # Nếu tokenizer được cung cấp và có embedding matrix, khởi tạo token_emb từ đó
        if exists(tokenizer) and hasattr(tokenizer, 'get_embedding_matrix'):
            pretrained_embeddings = tokenizer.get_embedding_matrix()
            # Chỉ sao chép embedding cho các token có trong từ điển (theo ID)
            with torch.no_grad():
                for token, idx in tokenizer.token2idx.items():
                    if idx < self.token_emb.weight.shape[0]:
                        self.token_emb.weight[idx] = torch.tensor(pretrained_embeddings[idx], dtype=self.token_emb.weight.dtype)
            # Lưu thông báo
            print("Đã khởi tạo token embeddings từ tokenizer với các vector đã căn chỉnh.")
            
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, dim) * 0.02)
        self.text_encoder = nn.Sequential(
            nn.LayerNorm(dim),
            Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ),
            nn.LayerNorm(dim)
        )
        
        # Thêm Cross-Region Attention để kết nối image-text regions
        self.cross_region_attention = CrossRegionAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Decoder model
        self.decoder = MambaDecoder(
            dim=dim,
            num_tokens=num_tokens,
            depth=decoder_depth,
            context_dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        
        # Áp dụng khởi tạo tốt hơn
        self.apply(self._initialize_parameters)
        
    def _initialize_parameters(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def encode_text(self, text):
        """Mã hóa text cho contrastive learning"""
        x = self.token_emb(text) + self.pos_emb[:, :text.shape[1]]
        return self.text_encoder(x)
    
    def create_abnormal_terms_mask(self, text, tokenizer):
        """Tạo mask cho các từ chỉ bất thường trong văn bản"""
        batch_size, seq_len = text.shape
        mask = torch.zeros(batch_size, seq_len, device=text.device)
        
        # Giải mã tokens
        decoded_texts = tokenizer.decode_batch(text.cpu().numpy())
        
        # Đánh dấu vị trí các từ bất thường
        for b, text_str in enumerate(decoded_texts):
            text_lower = text_str.lower()
            words = text_lower.split()
            
            for i, word in enumerate(words):
                if i < seq_len:
                    for term in self.abnormal_terms:
                        if term in word:
                            mask[b, i] = 1.0
        
        return mask
        
    def forward(
        self,
        text=None,
        images=None,
        tokenizer=None,
        return_loss=False,
        return_embedding=False,
        return_attention_maps=False
    ):
        b, device = text.shape[0], text.device
        
        # Image encoding
        if exists(self.img_encoder):
            image_embeddings = self.img_encoder(images)
        else:
            image_embeddings = images
        
        img_emb = self.img_proj(image_embeddings)  # [batch, 49, dim]
        
        # Áp dụng medical attention cho image features
        img_emb = self.img_med_attention(img_emb)
        
        # Text encoding - encodings đơn giản cho contrastive
        batch_size = img_emb.shape[0]
        text_emb = self.encode_text(text)  # [batch, seq_len, dim]
        
        # Áp dụng Cross-Region Attention giữa các vùng hình ảnh và tokens văn bản
        enhanced_img, enhanced_text, img_to_text_attn, text_to_img_attn = self.cross_region_attention(
            img_emb, text_emb
        )
        
        # Tạo mask cho các từ bất thường nếu có tokenizer
        abnormal_terms_mask = None
        if tokenizer is not None:
            abnormal_terms_mask = self.create_abnormal_terms_mask(text, tokenizer)
        
        if return_embedding:
            # Trả về embeddings sau cross-region attention
            img_emb_mean = enhanced_img.mean(dim=1)     # [batch, dim]
            text_emb_mean = enhanced_text.mean(dim=1)   # [batch, dim]
            
            # Normalize cho contrastive
            img_emb_norm = F.normalize(img_emb_mean, dim=-1)
            text_emb_norm = F.normalize(text_emb_mean, dim=-1)
            
            return img_emb_norm, text_emb_norm
            
        if return_attention_maps:
            # Trả về attention maps để phân tích và trực quan hóa
            # Detect abnormal regions if tokenizer provided
            abnormal_regions = None
            if tokenizer is not None:
                abnormal_regions = detect_abnormal_regions(
                    enhanced_img, text, text_to_img_attn, tokenizer, self.abnormal_terms
                )
            
            return {
                'img_to_text_attn': img_to_text_attn,  # [batch, heads, 49, seq_len]
                'text_to_img_attn': text_to_img_attn,  # [batch, heads, seq_len, 49]
                'abnormal_regions': abnormal_regions    # [batch, 7, 7] or None
            }
        
        if return_loss:
            # Mean pooling cho global contrastive
            img_emb_mean = enhanced_img.mean(dim=1)     # [batch, dim]
            text_emb_mean = enhanced_text.mean(dim=1)   # [batch, dim]
            
            # Normalize cho contrastive
            img_emb_norm = F.normalize(img_emb_mean, dim=-1)
            text_emb_norm = F.normalize(text_emb_mean, dim=-1)
            
            # Shifted right cho autoregressive training
            text_input = text[:, :-1]
            text_target = text[:, 1:]
            
            # Decoder với Cross-attention
            logits, _ = self.decoder(text_input, enhanced_img)
            
            # NLL loss cho captioning
            caption_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                text_target.reshape(-1),
                label_smoothing=0.1
            )
            
            # Global contrastive loss
            sim_i2t = img_emb_norm @ text_emb_norm.T * self.temperature
            sim_t2i = text_emb_norm @ img_emb_norm.T * self.temperature
            
            labels = torch.arange(batch_size, device=device)
            
            contrastive_loss_i2t = F.cross_entropy(sim_i2t, labels)
            contrastive_loss_t2i = F.cross_entropy(sim_t2i, labels)
            contrastive_loss = (contrastive_loss_i2t + contrastive_loss_t2i) / 2
            
            # Cross-Region Attention loss
            cra_loss = cross_region_loss(
                img_to_text_attn, text_to_img_attn, abnormal_terms_mask
            )
            
            # Dynamic caption weight dựa trên epoch
            dynamic_caption_weight = self.caption_loss_weight * min(1.5, 1.0 + 0.5 * self.epoch_tracker / self.max_epochs)
            
            # Cân bằng gradient tốt hơn
            loss_weights = {
                'contrastive': self.contrastive_loss_weight,
                'caption': dynamic_caption_weight,
                'cross_region': self.cross_region_loss_weight
            }
            
            # Chuẩn hóa weights tổng bằng 1
            weight_sum = sum(loss_weights.values()) + 1e-8
            loss_weights = {k: v/weight_sum for k, v in loss_weights.items()}
            
            # Tính loss với weights chuẩn hóa
            loss = (
                loss_weights['contrastive'] * contrastive_loss + 
                loss_weights['caption'] * caption_loss +
                loss_weights['cross_region'] * cra_loss
            )
            
            return loss
        
        # Inference mode
        logits, cross_attentions = self.decoder(text, enhanced_img)
        
        # Trả về logits và cross-attention nếu cần
        if return_attention_maps:
            return logits, cross_attentions
        
        return logits


def exists(val):
    return val is not None


def create_anatomamba_cra(
    *,
    image_size=224,
    in_chans=3,
    depths=[2, 2, 2, 2],
    dims=[64, 128, 256, 512],
    d_state=16,
    d_conv=4,
    expand=2,
    drop_path_rate=0.1,
    mamba_dim=512,
    num_tokens=10000,
    decoder_depth=6,
    caption_loss_weight=1.0,
    contrastive_loss_weight=1.0,
    cross_region_loss_weight=0.5,
    max_epochs=50,
    num_heads=8,
    abnormal_terms=None,
    tokenizer=None,
    **kwargs
):
    """Tạo mô hình AnatomaMamba với Cross-Region Attention"""

    # Tạo mô hình ConvNeXt-Mamba cho encoder
    img_encoder = create_convnext_mamba_coca(
        image_size=image_size,
        in_chans=in_chans,
        depths=depths,
        dims=dims,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    # Tạo mô hình AnatomaMamba with Cross-Region Attention
    model = AnatomaMambaCRA(
        dim=mamba_dim,
        img_encoder=img_encoder,
        num_tokens=num_tokens,
        decoder_depth=decoder_depth,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        caption_loss_weight=caption_loss_weight,
        contrastive_loss_weight=contrastive_loss_weight,
        cross_region_loss_weight=cross_region_loss_weight,
        max_epochs=max_epochs,
        num_heads=num_heads,
        abnormal_terms=abnormal_terms,
        tokenizer=tokenizer
    )
    
    return model


# Cấu hình mẫu - nhẹ hơn để chạy trên dataset nhỏ
small_config = {
    'image_size': 224,
    'in_chans': 3,
    'depths': [1, 1, 1, 1],
    'dims': [32, 64, 128, 256],
    'd_state': 8,
    'd_conv': 4,
    'expand': 1,
    'drop_path_rate': 0.1,
    'mamba_dim': 384,
    'num_tokens': 10000,
    'decoder_depth': 4,
    'caption_loss_weight': 1.0,
    'contrastive_loss_weight': 1.0,
    'cross_region_loss_weight': 0.5,
    'max_epochs': 50,
    'num_heads': 4
}

# Cấu hình trung bình - cân bằng giữa hiệu năng và kích thước
base_config = {
    'image_size': 224,
    'in_chans': 3,
    'depths': [2, 2, 2, 2],
    'dims': [64, 128, 256, 512],
    'd_state': 16,
    'd_conv': 4,
    'expand': 2,
    'drop_path_rate': 0.1,
    'mamba_dim': 512,
    'num_tokens': 10000,
    'decoder_depth': 6,
    'caption_loss_weight': 1.0,
    'contrastive_loss_weight': 1.0,
    'cross_region_loss_weight': 0.5,
    'max_epochs': 50,
    'num_heads': 8
} 