import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from modules.tokenizers import Tokenizer
from models.anatomamba_cra import create_anatomamba_cra, small_config
from config.configs import Test_Config
import os
import cv2

# Thiết lập các tham số
args = Test_Config()


def visualize_attention_map(img_path, report, model, tokenizer, device, save_dir='attention_maps'):
    """
    Trực quan hóa attention map giữa hình ảnh X-quang và báo cáo
    
    Parameters:
        img_path: Đường dẫn đến hình ảnh X-quang
        report: Chuỗi báo cáo y tế
        model: Mô hình AnatomaMambaCRA đã huấn luyện
        tokenizer: Tokenizer để xử lý văn bản
        device: Thiết bị tính toán (CPU/GPU)
        save_dir: Thư mục lưu kết quả
    """
    # Tạo thư mục lưu kết quả
    os.makedirs(save_dir, exist_ok=True)
    
    # Chuẩn bị hình ảnh
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Đọc và chuẩn bị hình ảnh
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Chuẩn bị văn bản báo cáo
    report_clean = tokenizer.clean_report(report)
    tokens = report_clean.split()
    token_ids = tokenizer(report)
    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    # Lấy attention maps
    model.eval()
    with torch.no_grad():
        attention_maps = model(
            text=token_ids_tensor,
            images=img_tensor,
            tokenizer=tokenizer,
            return_attention_maps=True
        )
    
    # Chuyển đổi hình ảnh cho visualization
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (224, 224))
    
    # Lấy các attention maps
    img_to_text_attn = attention_maps['img_to_text_attn']  # [1, heads, 49, seq_len]
    text_to_img_attn = attention_maps['text_to_img_attn']  # [1, heads, seq_len, 49]
    abnormal_regions = attention_maps['abnormal_regions']  # [1, 7, 7]
    
    # Trung bình trên các heads
    img_to_text_mean = img_to_text_attn.mean(dim=1)[0]  # [49, seq_len]
    text_to_img_mean = text_to_img_attn.mean(dim=1)[0]  # [seq_len, 49]
    
    # Visualization 1: Heatmap cho vùng bất thường
    if abnormal_regions is not None:
        abnormal_heatmap = abnormal_regions[0].cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img_resized)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_resized)
        abnormal_resized = cv2.resize(abnormal_heatmap, (224, 224))
        plt.imshow(abnormal_resized, alpha=0.6, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title('Abnormal Regions')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/abnormal_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Visualization 2: Attention từ text sang image cho một số từ quan trọng
    # Chọn các từ quan trọng từ báo cáo
    important_words = []
    abnormal_terms = [
        'opacity', 'mass', 'nodule', 'abnormal', 'effusion', 'pneumonia',
        'cardiomegaly', 'edema', 'atelectasis', 'consolidation', 'lesion',
        'normal', 'clear', 'unchanged', 'stable'
    ]
    
    word_indices = []
    for i, word in enumerate(tokens):
        if any(term in word.lower() for term in abnormal_terms):
            if i < len(token_ids) - 2:  # Trừ token bắt đầu và kết thúc
                important_words.append(word)
                word_indices.append(i + 1)  # +1 vì token bắt đầu
    
    if not important_words:
        # Nếu không tìm thấy từ đặc biệt, chọn một số từ bất kỳ
        indices = np.linspace(1, len(tokens) - 1, min(5, len(tokens) - 1)).astype(int)
        important_words = [tokens[i-1] for i in indices]
        word_indices = indices
    
    # Giới hạn số lượng từ để visualization
    max_words = min(5, len(important_words))
    important_words = important_words[:max_words]
    word_indices = word_indices[:max_words]
    
    # Visualization cho mỗi từ quan trọng
    plt.figure(figsize=(15, 3 * max_words))
    
    for i, (word, idx) in enumerate(zip(important_words, word_indices)):
        if idx >= text_to_img_mean.shape[0]:
            continue
            
        # Lấy attention map từ từ này đến hình ảnh
        attention = text_to_img_mean[idx].reshape(7, 7).cpu().numpy()
        attention_resized = cv2.resize(attention, (224, 224))
        
        plt.subplot(max_words, 2, i*2 + 1)
        plt.text(0.5, 0.5, word, fontsize=20, ha='center')
        plt.axis('off')
        
        plt.subplot(max_words, 2, i*2 + 2)
        plt.imshow(img_resized)
        plt.imshow(attention_resized, alpha=0.7, cmap='jet')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f'Attention: "{word}" -> Image')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/text_to_image_attention.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualization 3: Attention từ image sang text cho một số vùng quan trọng
    # Chọn các vùng có attention cao nhất
    region_importance = img_to_text_mean.sum(dim=1).cpu().numpy()
    top_regions = np.argsort(region_importance)[-5:]
    
    plt.figure(figsize=(15, 3 * len(top_regions)))
    
    for i, region_idx in enumerate(top_regions):
        # Tạo mặt nạ cho vùng này
        region_mask = np.zeros((7, 7))
        region_mask.flat[region_idx] = 1
        region_mask_resized = cv2.resize(region_mask, (224, 224))
        
        # Lấy attention từ vùng này đến text
        region_to_text = img_to_text_mean[region_idx].cpu().numpy()
        
        # Hiển thị top 10 từ có attention cao nhất
        top_token_indices = np.argsort(region_to_text)[-10:]
        top_tokens = [tokens[idx-1] if idx-1 < len(tokens) else "" for idx in top_token_indices]
        top_attentions = region_to_text[top_token_indices]
        
        plt.subplot(len(top_regions), 2, i*2 + 1)
        plt.imshow(img_resized)
        plt.imshow(region_mask_resized, alpha=0.7, cmap='jet')
        plt.title(f'Region {region_idx}')
        plt.axis('off')
        
        plt.subplot(len(top_regions), 2, i*2 + 2)
        plt.bar(range(len(top_tokens)), top_attentions)
        plt.xticks(range(len(top_tokens)), top_tokens, rotation=45, ha='right')
        plt.title(f'Top words attended by Region {region_idx}')
        plt.tight_layout()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/image_to_text_attention.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention maps saved to {save_dir}!")
    return attention_maps


def load_model_for_visualization(checkpoint_path, device):
    """
    Tải mô hình để trực quan hóa
    """
    # Tạo tokenizer
    tokenizer = Tokenizer(args)
    
    # Tạo mô hình
    model_config = small_config.copy()
    model = create_anatomamba_cra(**model_config)
    
    # Tải checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, tokenizer


def main():
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Đường dẫn đến checkpoint
    checkpoint_path = args.load  # Sử dụng checkpoint từ Test_Config
    
    # Tải mô hình và tokenizer
    model, tokenizer = load_model_for_visualization(checkpoint_path, device)
    
    # Đường dẫn đến hình ảnh và báo cáo mẫu
    img_path = os.path.join(args.image_dir, "CXR1_1_IM-0001-3001.png")  # Thay đổi theo dataset
    
    # Báo cáo mẫu (có thể thay đổi)
    report = """
    The heart size and pulmonary vascularity appear within normal limits. 
    There is no focal consolidation, pneumothorax, or pleural effusion. 
    There are degenerative changes in the thoracic spine.
    """
    
    # Trực quan hóa
    attention_maps = visualize_attention_map(
        img_path=img_path,
        report=report,
        model=model,
        tokenizer=tokenizer,
        device=device,
        save_dir='attention_visualization'
    )


if __name__ == "__main__":
    main() 