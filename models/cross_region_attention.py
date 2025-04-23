import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossRegionAttention(nn.Module):
    """
    Cross-Region Attention: Học mối quan hệ hai chiều giữa các vùng hình ảnh (7x7) và tokens văn bản
    
    Cơ chế này tạo ra sự tương tác hai chiều:
    1. Image -> Text: Mỗi vùng hình ảnh chú ý đến tokens văn bản phù hợp
    2. Text -> Image: Mỗi token văn bản chú ý đến vùng hình ảnh tương ứng
    
    Input:
        - img_regions: [batch, num_regions, dim] (thường là [batch, 49, dim] với 7x7 regions)
        - text_tokens: [batch, seq_len, dim]
        
    Output:
        - enhanced_img_regions: [batch, num_regions, dim]
        - enhanced_text_tokens: [batch, seq_len, dim]
        - img_to_text_attn: Attention weights từ hình ảnh đến văn bản
        - text_to_img_attn: Attention weights từ văn bản đến hình ảnh
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Adaptive temperature parameter thay vì scale cố định
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
        # Projections cho image regions với bias
        self.img_q = nn.Linear(dim, dim, bias=True)
        self.img_k = nn.Linear(dim, dim, bias=True)
        self.img_v = nn.Linear(dim, dim, bias=True)
        
        # Projections cho text tokens với bias
        self.text_q = nn.Linear(dim, dim, bias=True)
        self.text_k = nn.Linear(dim, dim, bias=True)
        self.text_v = nn.Linear(dim, dim, bias=True)
        
        # Output projections
        self.img_out = nn.Linear(dim, dim)
        self.text_out = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Cải tiến Gating mechanism với sigmoid+tanh
        self.img_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        self.text_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        
        # Layer norms
        self.img_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)
        
        # Thêm residual projection để tăng gradient flow
        self.img_residual = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.text_residual = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        
        # Thêm positional bias cho từng vùng
        self.region_pos_bias = nn.Parameter(torch.zeros(1, 49, 1))
        self.token_pos_bias = nn.Parameter(torch.zeros(1, 1, 100))  # Giả sử max length = 100
        
    def _reshape_for_attention(self, x, batch_size):
        """Reshape tensor để thực hiện multi-head attention"""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
    def forward(self, img_regions, text_tokens):
        batch_size = img_regions.shape[0]
        num_regions = img_regions.shape[1]
        seq_len = text_tokens.shape[1]
        
        # Normalize inputs
        img_regions_norm = self.img_norm(img_regions)
        text_tokens_norm = self.text_norm(text_tokens)
        
        # Image projections
        img_q = self._reshape_for_attention(self.img_q(img_regions_norm), batch_size)
        img_k = self._reshape_for_attention(self.img_k(img_regions_norm), batch_size)
        img_v = self._reshape_for_attention(self.img_v(img_regions_norm), batch_size)
        
        # Text projections
        text_q = self._reshape_for_attention(self.text_q(text_tokens_norm), batch_size)
        text_k = self._reshape_for_attention(self.text_k(text_tokens_norm), batch_size)
        text_v = self._reshape_for_attention(self.text_v(text_tokens_norm), batch_size)
        
        # Adaptive temperature
        scale = self.temperature.clamp(min=0.01)
        
        # Cross-attention: Image -> Text với positional bias
        img_text_attn = torch.matmul(img_q, text_k.transpose(-1, -2)) / scale
        
        # Tạo positional bias cho tokens, đảm bảo kích thước phù hợp
        # Sử dụng broadcast tự nhiên thay vì reshape không chính xác
        token_bias = self.token_pos_bias[:, :, :seq_len]  # [1, 1, seq_len]
        # Thêm bias không phụ thuộc vào cấu trúc multi-head
        img_text_attn = img_text_attn + token_bias
        
        img_text_attn = F.softmax(img_text_attn, dim=-1)
        img_text_attn = self.dropout(img_text_attn)
        img_attended_text = torch.matmul(img_text_attn, text_v)
        
        # Cross-attention: Text -> Image với positional bias
        text_img_attn = torch.matmul(text_q, img_k.transpose(-1, -2)) / scale
        
        # Tạo positional bias cho regions, đảm bảo kích thước phù hợp
        region_bias = self.region_pos_bias  # [1, num_regions, 1]
        # Thêm bias không phụ thuộc vào cấu trúc multi-head
        # Chuyển đổi để phù hợp với kích thước của text_img_attn [batch, heads, seq_len, regions]
        text_img_attn = text_img_attn + region_bias.transpose(-1, -2)
        
        text_img_attn = F.softmax(text_img_attn, dim=-1)
        text_img_attn = self.dropout(text_img_attn)
        text_attended_img = torch.matmul(text_img_attn, img_v)
        
        # Reshape lại
        img_attended_text = img_attended_text.transpose(1, 2).reshape(
            batch_size, num_regions, -1)
        text_attended_img = text_attended_img.transpose(1, 2).reshape(
            batch_size, seq_len, -1)
        
        # Gating mechanism cải tiến với residual
        img_gate_input = torch.cat([img_regions, img_attended_text], dim=-1)
        text_gate_input = torch.cat([text_tokens, text_attended_img], dim=-1)
        
        img_gate_value = self.img_gate(img_gate_input)
        text_gate_value = self.text_gate(text_gate_input)
        
        # Áp dụng gate và projection với residual connections
        img_residual = self.img_residual(img_regions)
        text_residual = self.text_residual(text_tokens)
        
        enhanced_img = img_residual + img_gate_value * self.img_out(img_attended_text)
        enhanced_text = text_residual + text_gate_value * self.text_out(text_attended_img)
        
        # Trả về kết quả và attention maps để phân tích
        return enhanced_img, enhanced_text, img_text_attn, text_img_attn
    
    
def cross_region_loss(img_to_text_attn, text_to_img_attn, abnormal_terms_mask=None, anatomical_regions_mask=None):
    """
    Tính loss đặc biệt cho Cross-Region Attention với nhiều thành phần hơn
    
    Parameters:
        - img_to_text_attn: [batch, heads, regions, seq_len]
        - text_to_img_attn: [batch, heads, seq_len, regions]
        - abnormal_terms_mask: [batch, seq_len] - Mask cho các từ chỉ bất thường (optional)
        - anatomical_regions_mask: [batch, regions] - Mask cho các vùng giải phẫu quan trọng (optional)
    
    Returns:
        - Tổng hợp các thành phần loss
    """
    batch_size = img_to_text_attn.shape[0]
    
    # Giảm chiều num_heads
    img_to_text = img_to_text_attn.mean(dim=1)  # [batch, regions, seq_len]
    text_to_img = text_to_img_attn.mean(dim=1)  # [batch, seq_len, regions]
    
    # 1. Diversity loss với KL divergence để khuyến khích phân phối đa dạng hơn
    # Tính entropy của phân phối attention - entropy cao = đa dạng
    # Uniform distribution reference
    uniform_text = torch.ones_like(img_to_text) / img_to_text.size(-1)
    uniform_img = torch.ones_like(text_to_img) / text_to_img.size(-1)
    
    # KL divergence để đo sự khác biệt so với phân phối đồng nhất
    kl_img = F.kl_div(
        F.log_softmax(img_to_text, dim=-1),
        uniform_text,
        reduction='none'
    ).sum(-1).mean()
    
    kl_text = F.kl_div(
        F.log_softmax(text_to_img, dim=-1),
        uniform_img,
        reduction='none'
    ).sum(-1).mean()
    
    # Inverse KL để khuyến khích entropy cao (đa dạng)
    diversity_loss = -0.5 * (kl_img + kl_text)
    
    # 2. Coherence loss: Khuyến khích sự nhất quán giữa hai hướng attention
    # Sử dụng cosine similarity để đo sự tương đồng tốt hơn
    text_to_img_t = text_to_img.transpose(-1, -2)  # [batch, regions, seq_len]
    
    # Chuẩn hóa để sử dụng cosine similarity
    img_to_text_norm = F.normalize(img_to_text, p=2, dim=-1)
    text_to_img_t_norm = F.normalize(text_to_img_t, p=2, dim=-1)
    
    # 1 - cosine_similarity để có loss (nhỏ khi giống nhau)
    coherence_loss = 1 - (img_to_text_norm * text_to_img_t_norm).sum(-1).mean()
    
    # 3. Abnormality focus loss (nếu có mask abnormal terms)
    abnormality_loss = 0
    if abnormal_terms_mask is not None:
        # Tăng attention cho vùng văn bản chứa từ bất thường 
        # abnormal_terms_mask: [batch, seq_len]
        abnormal_mask = abnormal_terms_mask.unsqueeze(1)  # [batch, 1, seq_len]
        
        # Top-k pooling để tập trung vào vùng có attention cao nhất
        # Sắp xếp attention scores
        topk = max(1, int(img_to_text.size(-1) * 0.1))  # Top 10% tokens
        img_to_text_sorted, _ = torch.sort(img_to_text, dim=-1, descending=True)
        topk_mean = img_to_text_sorted[:, :, :topk].mean(dim=-1)  # [batch, regions]
        
        # Tập trung vào vùng bất thường
        abnormal_attention = (img_to_text * abnormal_mask).sum(dim=-1) / (abnormal_mask.sum(dim=-1) + 1e-10)
        
        # So sánh attention của từ bất thường với top-k mean
        # Từ bất thường nên có attention cao hơn trung bình
        abnormality_loss = F.relu(topk_mean - abnormal_attention).mean()
    
    # 4. Anatomical region focus (nếu có mask cho vùng giải phẫu)
    anatomical_loss = 0
    if anatomical_regions_mask is not None:
        # anatomical_regions_mask: [batch, regions]
        # Khuyến khích attention từ text đến vùng giải phẫu quan trọng
        region_importance = anatomical_regions_mask.unsqueeze(1)  # [batch, 1, regions]
        
        # Trọng số attention cao hơn cho vùng quan trọng
        weighted_attention = text_to_img * region_importance
        anatomical_loss = -weighted_attention.mean()
    
    # 5. Focal attention loss - khuyến khích attention mạnh hơn cho các vùng quan trọng
    gamma = 2.0  # Focal loss parameter
    focal_img = -(1 - img_to_text).pow(gamma) * torch.log(img_to_text + 1e-10)
    focal_text = -(1 - text_to_img).pow(gamma) * torch.log(text_to_img + 1e-10)
    focal_loss = 0.1 * (focal_img.mean() + focal_text.mean())
    
    # Tổng hợp loss với các hệ số cân bằng điều chỉnh
    total_loss = (
        diversity_loss * 0.1 +       # Khuyến khích đa dạng vừa phải
        coherence_loss * 1.0 +       # Nhất quán giữa 2 hướng quan trọng 
        abnormality_loss * 2.0 +     # Tăng cường chú ý đến bất thường
        anatomical_loss * 1.5 +      # Chú ý đến vùng giải phẫu quan trọng
        focal_loss * 0.5             # Tập trung vào vùng quan trọng
    )
    
    return total_loss


def detect_abnormal_regions(img_regions, text_tokens, text_to_img_attn, tokenizer, abnormal_terms):
    """
    Phát hiện các vùng bất thường dựa trên attention từ từ bất thường đến vùng ảnh
    với thuật toán cải tiến để xác định chính xác hơn
    
    Parameters:
        - img_regions: [batch, regions, dim]
        - text_tokens: [batch, seq_len]
        - text_to_img_attn: [batch, heads, seq_len, regions]
        - tokenizer: Tokenizer để giải mã tokens
        - abnormal_terms: List các từ chỉ bất thường
        
    Returns:
        - abnormal_heatmap: [batch, 7, 7] - Heatmap cho các vùng bất thường
        - abnormal_mask: [batch, seq_len] - Mask các từ bất thường được phát hiện
    """
    batch_size = text_tokens.shape[0]
    seq_len = text_tokens.shape[1]
    
    # Khởi tạo mask cho từ bất thường
    abnormal_mask = torch.zeros(batch_size, seq_len, device=text_tokens.device)
    
    # Khởi tạo dictionary để lưu trữ các vùng bất thường đã phát hiện cho mỗi mẫu
    abnormal_regions_dict = {}
    
    # Giải mã text tokens
    decoded_texts = tokenizer.decode_batch(text_tokens.cpu().numpy())
    
    # Tạo danh sách từ bất thường mở rộng với biến thể (đơn giản)
    extended_abnormal_terms = set()
    for term in abnormal_terms:
        extended_abnormal_terms.add(term)
        extended_abnormal_terms.add(term + 's')  # Thêm dạng số nhiều
        extended_abnormal_terms.add(term + 'es')  # Thêm dạng số nhiều khác
        # Thêm một số tiền tố/hậu tố thông dụng
        extended_abnormal_terms.add(term + 'ity')
        extended_abnormal_terms.add('ab' + term)
        extended_abnormal_terms.add('un' + term)
    
    # Với mỗi batch, đánh dấu chính xác các từ bất thường
    for b, text in enumerate(decoded_texts):
        # Áp dụng preprocessing đơn giản
        text = text.lower()
        words = text.split()
        
        # Phát hiện vị trí từ bất thường
        abnormal_positions = []
        
        # Xét từng từ trong văn bản
        for i, word in enumerate(words):
            if i >= seq_len:
                break
                
            # Kiểm tra nếu từ này là từ bất thường
            for term in extended_abnormal_terms:
                if term in word or word in term:
                    abnormal_mask[b, i] = 1.0
                    abnormal_positions.append(i)
                    break
        
        # Phát hiện vùng lân cận từ bất thường (context window)
        for i in abnormal_positions:
            # Add context window (trước/sau 2 từ)
            start_idx = max(0, i - 2)
            end_idx = min(seq_len, i + 3)
            abnormal_mask[b, start_idx:end_idx] = abnormal_mask[b, start_idx:end_idx] * 0.7 + 0.3
        
        # Lưu vị trí từ bất thường cho batch này
        abnormal_regions_dict[b] = abnormal_positions
    
    # Trung bình attention của các heads
    text_to_img_mean = text_to_img_attn.mean(dim=1)  # [batch, seq_len, regions]
    
    # Ensemble nhiều heads thay vì chỉ lấy trung bình
    # Lấy top-k heads có attention cao nhất
    num_heads = text_to_img_attn.size(1)
    topk_heads = min(3, num_heads)  # Sử dụng top-3 heads
    
    # Tính toán attention trung bình từ top-k heads
    head_attention_sum = torch.sum(text_to_img_attn, dim=-1)  # [batch, heads, seq_len]
    _, top_head_indices = torch.topk(head_attention_sum, topk_heads, dim=1)
    
    # Tạo mask cho top-k heads
    head_mask = torch.zeros(batch_size, num_heads, 1, 1, device=text_to_img_attn.device)
    for b in range(batch_size):
        head_mask[b, top_head_indices[b], :, :] = 1.0
    
    # Áp dụng mask và lấy trung bình
    top_heads_attention = (text_to_img_attn * head_mask).sum(dim=1) / topk_heads
    
    # Lấy cả attention trung bình và top-k
    alpha = 0.7  # Trọng số kết hợp
    ensemble_attention = alpha * top_heads_attention + (1 - alpha) * text_to_img_mean
    
    # Áp dụng abnormal mask để tạo heatmap
    abnormal_mask_expanded = abnormal_mask.unsqueeze(-1)  # [batch, seq_len, 1]
    weighted_attention = ensemble_attention * abnormal_mask_expanded
    
    # Tổng hợp attention dựa trên abnormal mask
    sum_mask = abnormal_mask.sum(dim=1, keepdim=True) + 1e-10
    abnormal_attention = weighted_attention.sum(dim=1) / sum_mask  # [batch, regions]
    
    # Áp dụng spatial smoothing để làm mịn heatmap
    abnormal_attention = abnormal_attention.reshape(batch_size, 7, 7)
    
    # Tăng cường contrast bằng min-max normalization
    min_vals = abnormal_attention.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
    max_vals = abnormal_attention.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
    normalized_attention = (abnormal_attention - min_vals) / (max_vals - min_vals + 1e-10)
    
    # Áp dụng power normalization để tăng cường contrast
    gamma = 0.5  # Giá trị nhỏ hơn 1 làm nổi bật vùng có attention thấp
    abnormal_heatmap = normalized_attention.pow(gamma)
    
    return abnormal_heatmap, abnormal_mask 