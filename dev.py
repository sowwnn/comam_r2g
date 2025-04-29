import torch
import numpy as np
from tqdm import tqdm
from modules.dataloaders import R2DataLoader
from modules.tokenizers_enhanced import EnhancedTokenizer as Tokenizer
import os
import json

from config.configs import Test_Config

# from vit_pytorch.extractor import Extractor
from coca_pytorch.coca_pytorch import CoCa
# from models.coca import CoCa
from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

# from vit_pytorch.simple_vit_with_patch_dropout import SimpleViT
# from vit_pytorch.cross_vit import CrossViT
# from modules.crossvit import eCrossViT
# from models.adapt_vit import AdaptiveRegionViT
# from models.cnext_mam import create_convnext_mamba_coca
# from models.cnext_mam_2_org import create_convnext_mamba_coca
# from models.cnext_mam_threshold import create_convnext_mamba_threshold
from models.cnext_mam_3 import create_convnext_mamba_coca
# from models.comamba import create_comamba
from models.anatomamba import create_anatomamba, small_config, base_config
from models.anatomamba_cra import create_anatomamba_cra

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=5401):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_model_config():
    """
    Trả về cấu hình mô hình AnatomaMamba siêu nhẹ cho dataset nhỏ
    """
    # Sử dụng config từ anatomamba.py và điều chỉnh để nhẹ hơn
    config = base_config.copy()
    
    # Giảm kích thước thêm nữa nếu cần
    config['depths'] = [1, 1, 1, 1]
    config['dims'] = [64, 128, 256, 512]
    config['d_state'] = 8
    config['mamba_dim'] = 512
    config['decoder_depth'] = 8
    config['num_heads'] = 8

    config['caption_loss_weight'] = 0.5
    config['contrastive_loss_weight'] = 0.3
    config['cross_region_loss_weight'] = 0.2

    config['text_layers'] = 8
    config['use_coca_text_encoder'] = True  # Sử dụng text encoder kiểu CoCa
    
    return config


def build_model(args, tokenizer, config):
    """
    Xây dựng mô hình AnatomaMamba với Cross-Region Attention
    """

    # abnormal_terms = [
    #     'opacity', 'mass', 'nodule', 'abnormal', 'effusion', 'pneumonia',
    #     'cardiomegaly', 'edema', 'atelectasis', 'consolidation', 'lesion'
    # ]
    
    abnormal_terms = [
        'pneumothorax', 'effusion', 'pleural effusion', 'consolidation', 'opacity', 'opacities',
        'pneumonia', 'pulmonary edema', 'edema', 'cardiomegaly', 'atelectasis', 'infiltrate',
        'nodule', 'fracture', 'degenerative', 'thoracic', 'granuloma', 'calcified', 
        'atherosclerotic', 'hiatal hernia', 'mass', 'lesion', 'fibrosis', 'infection',
        'calcification', 'hyperinflated', 'emphysema', 'spondylosis'
    ]
    
    # Thêm các tham số đặc thù cho Cross-Region Attention nếu chưa có
    print(config)
    
    # Tạo mô hình với Cross-Region Attention
    model = create_anatomamba_cra(
        image_size=config['image_size'],
        in_chans=config['in_chans'],
        depths=config['depths'],
        dims=config['dims'],
        d_state=config['d_state'],
        d_conv=config['d_conv'],
        expand=config['expand'],
        drop_path_rate=config['drop_path_rate'],
        mamba_dim=config['mamba_dim'],
        num_tokens=tokenizer.get_vocab_size() + 4,
        decoder_depth=config['decoder_depth'],
        caption_loss_weight=config['caption_loss_weight'],
        contrastive_loss_weight=config['contrastive_loss_weight'],
        cross_region_loss_weight=config['cross_region_loss_weight'],
        max_epochs=config['max_epochs'],
        num_heads=config['num_heads'],
        abnormal_terms=abnormal_terms,
        tokenizer=tokenizer,
        text_layers=config['text_layers']
    )

    # Lưu cấu hình mô hình
    config_path = os.path.join(args.save_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


    # weight = torch.load(args.load)
    # model.load_state_dict(weight['model_state_dict'])

    print("Build model AnatomaMamba with Cross-Region Attention successfully")
    return model


def count_parameters(model):
    """
    Đếm và hiển thị số lượng tham số của mô hình, phân loại theo từng module
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số: {total_params:,} (~{total_params/1e6:.2f}M)")
    
    # Đếm tham số theo từng module chính
    modules_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        modules_params[name] = params
    
    print("\nPhân bố tham số theo module:")
    for name, params in modules_params.items():
        print(f"{name}: {params:,} tham số ({params/total_params*100:.2f}%)")
    
    return total_params


def train_epoch(model, dataloader, scaler, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    
    # Cập nhật epoch hiện tại cho model
    if hasattr(model, 'epoch_tracker'):
        model.epoch_tracker = epoch
    
    for idx, batch in enumerate(dataloader):
        images = batch[1].to(device)
        text = batch[2].to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            loss = model(
                text=text,
                images=images,
                return_loss=True,
                # gradient_checkpointing=(len(dataloader) > 100)  # Chỉ áp dụng cho dataset lớn
            )
        
        # Gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        # scheduler.step(total_loss) 
        # current_lr = scheduler.get_last_lr()[0]
        # print(loss.item())
        
    return total_loss / len(dataloader)


def validate(model, dataloader, scheduler, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[1].to(device)
            text = batch[2].to(device)
            
            loss = model(
                text=text,
                images=images,
                return_loss=True
            )
            
            total_loss += loss.item()
    val_loss = total_loss / len(dataloader)
    return val_loss


def main(args):
    # Thiết lập seed
    set_seed()
    
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Tạo tokenizer
    tokenizer = Tokenizer(args)
    
    # Tạo dataloaders
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)

    # Khởi tạo GradScaler cho mixed precision
    scaler = GradScaler()

    # Tạo thư mục lưu mô hình
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Xây dựng mô hình
    model_config = get_model_config()
    model = build_model(args, tokenizer, model_config)
    
    # Đếm và hiển thị số lượng tham số
    total_params = count_parameters(model)
    print(f"\nKích thước mô hình: {total_params/1e6:.2f}M tham số")
    
    model = model.to(device)
    
    # Thiết lập optimizer
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), 
    #     lr=args.learning_rate, 
    #     weight_decay=0.05,
    #     betas=(0.9, 0.999)
    # )
    optimizer = Lion(
        model.parameters(),
        lr=1e-4,           # Sử dụng LR thấp hơn AdamW 3x
        weight_decay=1e-2,
        betas=(0.9, 0.99)
    )

    # Thiết lập scheduler với warmup
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_dataloader), eta_min=1e-6)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        pct_start=0.1,  # 10% đầu tiên là warmup
        total_steps=args.num_epochs, # * len(train_dataloader),
        anneal_strategy='cos',
        div_factor=25,  # LR ban đầu = max_lr/25
        final_div_factor=1000  # LR cuối = max_lr/1000
    )

    # Vòng lặp huấn luyện
    num_epochs = args.num_epochs
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Huấn luyện
        train_loss = train_epoch(model, train_dataloader, scaler, optimizer, scheduler, device, epoch)
        scheduler.step()
        # Đánh giá
        val_loss = validate(model, val_dataloader, scheduler, device)
        
        # Hiển thị thông tin
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Dynamic caption weight trong mô hình
        # if hasattr(model, 'caption_loss_weight'):
        #     dynamic_caption_weight = model.caption_loss_weight * min(1.5, 1.0 + 0.5 * epoch / num_epochs)
        #     print(f'Caption Loss Weight: {dynamic_caption_weight:.4f}')
        
        # Lưu mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'{args.save_dir}/best_model.pth')
            print(f'Saved new best model with validation loss: {val_loss:.4f}')
        
        # Luôn lưu mô hình mới nhất
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, f'{args.save_dir}/lastest.pth')


if __name__ == '__main__':
    args = Test_Config()
    main(args)
