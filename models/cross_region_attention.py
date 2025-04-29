import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
import json
import os


class CrossRegionAttention(nn.Module):
    def __init__(
        self, 
        dim, 
        img_size=7, 
        add_positional=True, 
        region_ann_path=None, 
        num_heads=8, 
        dropout=0.1,
        default_tokens=[2245, 4452, 1333],
        num_tokens=None
    ):
        super().__init__()
        self.norm_img = nn.LayerNorm(dim)
        self.norm_text = nn.Sequential(nn.Embedding(num_embeddings=num_tokens, embedding_dim=dim), nn.LayerNorm(dim))
        self.scale = dim ** -0.5
        self.img_size = img_size
        self.add_positional = add_positional
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Số lượng bộ phận cố định
        self.num_parts = 7
        
        # Hardcode 3 token cơ bản: [effusion, pleural, cardiomediastinal]
        # self.default_token_ids = 
        
        # Nếu có default_tokens được truyền vào, sử dụng chúng thay thế
        self.default_token_ids = default_tokens
            
        print(f"Sử dụng default tokens: {self.default_token_ids}")
        
        # Lưu trữ anatomical token groups từ file
        self.region_ann_path = region_ann_path
        self.anatomical_token_groups = {}
        self.token_to_group_map = {}
        self.load_anatomical_token_groups()
        
        # Positional embedding cho các vùng ảnh (7x7 grid)
        if add_positional:
            self.pos_embedding = nn.Parameter(torch.randn(img_size * img_size, dim // 4))
            self.pos_proj = nn.Linear(dim // 4, dim)
        
        # Self-attention cho image khi không có text
        self.img_self_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        
        # Projection heads cho contrastive learning
        self.img_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
    
    def add_spatial_info(self, img_features):
        batch, num_regions, dim = img_features.shape
        
        if not self.add_positional:
            return img_features
            
        # Thêm thông tin vị trí không gian
        pos_emb = self.pos_proj(self.pos_embedding) # [49, dim]
        
        # Thêm vào image features
        return img_features + pos_emb.unsqueeze(0)
    

    def load_anatomical_token_groups(self):
        """Đọc file anatomical_token_groups.json nếu được cung cấp"""
        self.anatomical_token_groups = {}
        self.token_to_group_map = {}
        
        try:
            # Đọc file JSON
            with open(self.region_ann_path, 'r') as f:
                groups_data = json.load(f)
                
            self.anatomical_token_groups = {int(k): v for k, v in groups_data.items()}
            for group_id, tokens in self.anatomical_token_groups.items():
                for token in tokens:
                    self.token_to_group_map[token] = group_id
                
            print(f"Đã tải {len(self.anatomical_token_groups)} nhóm token anatomical")
            
        except Exception as e:
            print(f"Lỗi khi tải anatomical_token_groups: {e}")
            self.anatomical_token_groups = {}
            self.token_to_group_map = {}
    
    def get_anatomical_parts(self, text_tokens, text_token_ids=None):
        """
        Trích xuất đúng 7 token IDs đại diện cho các bộ phận giải phẫu - đơn giản
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Khởi tạo tensor kết quả với giá trị 0
        part_token_ids = torch.zeros(batch_size, self.num_parts, dtype=torch.long, device=device)
        
        # Trường hợp 1: Không có text_token_ids
        if text_token_ids is None:
            # Điền default tokens
            num_default = min(len(self.default_token_ids), 3)
            for i in range(num_default):
                part_token_ids[:, i] = self.default_token_ids[i]
            return part_token_ids
        
        # Trường hợp 2: Có text_token_ids
        for b in range(batch_size):
            # Tìm tất cả group_ids xuất hiện trong batch này
            found_groups = []
            
            for tid in text_token_ids[b].cpu().tolist():
                # Token là key của một nhóm
                if tid in self.anatomical_token_groups and tid not in found_groups:
                    found_groups.append(tid)
                
                # Token thuộc một nhóm
                elif tid in self.token_to_group_map:
                    group_id = self.token_to_group_map[tid]
                    if group_id not in found_groups:
                        found_groups.append(group_id)
            
            # Lấy tối đa self.num_parts nhóm đầu tiên
            num_found = min(len(found_groups), self.num_parts)
            for i in range(num_found):
                part_token_ids[b, i] = found_groups[i]
        
        return part_token_ids    


    def forward(self, img_regions, text_tokens=None, text_token_ids=None):
        """
        Cross-Region Attention giữa image regions và 7 bộ phận giải phẫu
        """
        # Chuẩn hóa image features
        img_norm = self.norm_img(img_regions)  # [batch, 49, dim]
        img_spatial = self.add_spatial_info(img_norm)
        
        # Lấy 7 bộ phận giải phẫu từ text
        part_embeddings = self.get_anatomical_parts(text_tokens, text_token_ids)
        
        # Fallback: nếu không có part embeddings -> self-attention cho image
        if part_embeddings is None:
            enhanced_img, _ = self.img_self_attn(img_spatial, img_spatial, img_spatial)
            enhanced_img = img_regions + enhanced_img
            img_embed = self.img_proj(enhanced_img.mean(dim=1))
            return enhanced_img, None, img_embed, None, None, None
        
        # Chuẩn hóa part embeddings
        part_norm = self.norm_text(part_embeddings)  # [batch, 7, dim]
        
        # Cross-attention: image regions query part embeddings
        attn_scores = torch.bmm(img_spatial, part_norm.transpose(1, 2)) * self.scale  # [batch, 49, 7]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, 49, 7]
        
        # Tạo context từ part embeddings cho từng region
        region_context = torch.bmm(attn_weights, part_norm)  # [batch, 49, dim]
        
        # Enhanced image features
        enhanced_img = img_regions + region_context
        
        # Enhanced text features (không thay đổi)
        enhanced_text = text_tokens
        
        # Tính attention từ image sang part (cho visualize)
        img_to_part_attn = attn_weights.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch, heads, 49, 7]
        
        # Tính attention từ part sang image (cho visualize)
        part_img_attn = torch.bmm(part_norm, img_spatial.transpose(1, 2)) * self.scale  # [batch, 7, 49]
        part_img_weights = F.softmax(part_img_attn, dim=-1)  # [batch, 7, 49]
        part_to_img_attn = part_img_weights.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch, heads, 7, 49]
        
        # Contrastive embeddings
        img_embed = self.img_proj(enhanced_img.mean(dim=1))
        part_embed = self.text_proj(part_norm.mean(dim=1))
        
        return enhanced_img, enhanced_text, img_embed, part_embed, img_to_part_attn, part_to_img_attn
        