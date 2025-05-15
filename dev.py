import torch
import numpy as np
from tqdm import tqdm
from modules.dataloaders import R2DataLoader
# from modules.tokenizers_enhanced import EnhancedTokenizer as Tokenizer
from modules.tokenizers import Tokenizer
import os
import json

from config.configs import Test_Config

from lion_pytorch import Lion
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

from models.cnext_mam_3 import create_convnext_mamba_coca
from models.gate_attention_contrastive import GateAttentionContrastive


import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=5401):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)



def build_model(args, tokenizer):

    encoder = create_convnext_mamba_coca(
        image_size=224,
        in_chans=3,
        depths=[1, 1, 1, 1],
        dims=[64, 128, 256, 512],
        d_state=16,
    )

    model = GateAttentionContrastive(
        dim = 512,                    
        img_encoder = encoder,
        image_dim = 512,             
        num_tokens = tokenizer.get_vocab_size() + 4,
        unimodal_depth = 8, 
        multimodal_depth = 4, 
        dim_head = 32,
        heads = 16,                    
        caption_loss_weight = 1.,
        contrastive_loss_weight = 1.,
    )

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
                return_loss=True
            )
        
        # Gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        scheduler.step() 
        current_lr = scheduler.get_last_lr()[0]
        
    return total_loss / len(dataloader)

def train_epoch_(model, dataloader, scaler, optimizer, scheduler, device, epoch):
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
                return_loss=True
            )
        
        # Gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        
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
    _dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)


    scaler = GradScaler()
    os.makedirs(args.save_dir, exist_ok=True)
    
    model = build_model(args, tokenizer)


    total_params = count_parameters(model)
    print(f"\nKích thước mô hình: {total_params/1e6:.2f}M tham số")
    
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    # optimizer = Lion(
    #     model.parameters(),
    #     lr=1e-4,           # Sử dụng LR thấp hơn AdamW 3x
    #     weight_decay=1e-2,
    #     betas=(0.9, 0.99)
    # )

    # Thiết lập scheduler với warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_dataloader), eta_min=1e-6)
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=1e-4,
    #     pct_start=0.1,  # 10% đầu tiên là warmup
    #     total_steps=args.num_epochs * len(train_dataloader),
    #     anneal_strategy='cos',
    #     div_factor=25,  # LR ban đầu = max_lr/25
    #     final_div_factor=1000  # LR cuối = max_lr/1000
    # )


    # checkpoint = torch.load(args.load, weights_only=True)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.to(device)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # best_val_loss = checkpoint['val_loss']
    # train_loss = checkpoint['train_loss']

    # Vòng lặp huấn luyện
    num_epochs = args.num_epochs
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Huấn luyện
        train_loss = train_epoch(model, train_dataloader, scaler, optimizer, scheduler, device, epoch)
        if (epoch+1) % 2 == 0:
            _ = train_epoch_(model, val_dataloader, scaler, optimizer, scheduler, device, epoch)
            _ = train_epoch_(model, _dataloader, scaler, optimizer, scheduler, device, epoch)
        # Đánh giá
        val_loss = validate(model, val_dataloader, scheduler, device)
        
        # Hiển thị thông tin
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
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
