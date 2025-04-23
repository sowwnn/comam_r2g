import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
import math

class DyT(nn.Module):
    """
    Dynamic Tanh (DyT): Thay thế cho Layer Normalization
    Hỗ trợ cả tensor 3D và 4D với các định dạng channels_first và channels_last
    """
    def __init__(self, dim, alpha=1.0, beta=0.5, data_format="channels_last"):
        super().__init__()
        self.dim = dim
        self.data_format = data_format
        
        # Alpha: Tham số điều khiển độ dốc của hàm tanh
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        
        # Beta: Hệ số điều khiển mức độ bảo toàn thông tin gốc
        self.beta = nn.Parameter(torch.ones(1) * beta)
        
        # Scale và shift tương tự gamma, beta trong LayerNorm
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        """
        Args:
            x: Tensor đầu vào, định dạng [B, C, H, W] hoặc [B, H, W, C] hoặc [B, L, C]
        """
        original_x = x
        
        # Xử lý định dạng cho tensor 4D
        if len(x.shape) == 4:
            if self.data_format == "channels_first":
                # [B, C, H, W] -> [B, H, W, C]
                x = x.permute(0, 2, 3, 1)
            
            # Áp dụng tanh với hệ số alpha
            x_transformed = torch.tanh(self.alpha * x)
            
            # Kết hợp thông tin gốc để bảo toàn gradient và thông tin chi tiết
            x = self.beta * x_transformed + (1 - self.beta) * x
            
            # Áp dụng scale và shift
            x = x * self.scale + self.shift
            
            # Trả về đúng định dạng ban đầu
            if self.data_format == "channels_first":
                x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
                
        # Xử lý tensor 3D [B, L, C] hoặc 2D [B, C]
        else:
            # Áp dụng tanh với hệ số alpha
            x_transformed = torch.tanh(self.alpha * x)
            
            # Kết hợp thông tin gốc
            x = self.beta * x_transformed + (1 - self.beta) * x
            
            # Áp dụng scale và shift
            if len(x.shape) == 3:
                x = x * self.scale.view(1, 1, -1) + self.shift.view(1, 1, -1)
            else:
                x = x * self.scale.view(1, -1) + self.shift.view(1, -1)
                
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = DyT(dim, data_format="channels_last")  # Thay thế LayerNorm
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return input + self.drop_path(x)

class ConvNeXtStage(nn.Module):
    def __init__(self, dim_in, dim_out, depth, drop_path=0.):
        super().__init__()
        
        if dim_in != dim_out:
            self.downsample = nn.Sequential(
                DyT(dim_in, data_format="channels_first"),  # Thay thế LayerNorm
                nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2),
            )
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(dim_out, drop_path=drop_path)
            for _ in range(depth)
        ])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = DyT(dim)  # Thay thế LayerNorm
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x):
        # Convert 2D to 1D
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Process through Mamba
        x = self.norm(x)
        x = self.mamba(x)
        
        # Reshape back to 2D for easy return - will be useful for pyramid decoder
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class LateralConnection(nn.Module):
    """Lateral connection for Feature Pyramid Network"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        self.norm = DyT(out_dim, data_format="channels_last")  # Thay thế LayerNorm
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x

class ScaleAttention(nn.Module):
    """Attention mechanism for scale-adaptive fusion"""
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(dim, dim // 4, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(dim // 4, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        att = self.pool(x)
        att = self.fc1(att)
        att = self.act(att)
        att = self.fc2(att)
        att = self.sigmoid(att)
        return x * att

class PyramidFeatureDecoder(nn.Module):
    """Scale-Adaptive Pyramid Feature Decoder"""
    def __init__(self, dims, target_dim):
        super().__init__()
        
        # Lateral connections (to match dimensions)
        self.lateral_convs = nn.ModuleList([
            LateralConnection(dim, target_dim) 
            for dim in dims
        ])
        
        # Top-down pathway
        self.upsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(target_dim, target_dim, kernel_size=3, padding=1),
                DyT(target_dim, data_format="channels_first"),  # Thay thế LayerNorm
                nn.GELU()
            ) for _ in range(len(dims) - 1)
        ])
        
        # Scale attention for each level
        self.scale_attentions = nn.ModuleList([
            ScaleAttention(target_dim) for _ in range(len(dims))
        ])
        
        # Dynamic fusion gates
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(target_dim * 2, 2, kernel_size=1),
                nn.Sigmoid()
            ) for _ in range(len(dims) - 1)
        ])
        
        # Final output conv
        self.output_conv = nn.Sequential(
            nn.Conv2d(target_dim, target_dim, kernel_size=3, padding=1),
            DyT(target_dim, data_format="channels_first"),  # Thay thế LayerNorm
            nn.GELU()
        )
        
        # Target spatial size for output
        self.target_size = (7, 7)  # Default smallest feature map size
        
    def forward(self, features):
        # Process features through lateral connections
        laterals = [lateral_conv(feature) for lateral_conv, feature in zip(self.lateral_convs, features)]
        
        # Apply scale attention to each level
        attended_laterals = [attn(lateral) for attn, lateral in zip(self.scale_attentions, laterals)]
        
        # Top-down pathway with dynamic fusion
        pyramid_features = [attended_laterals[-1]]  # Start with the deepest level
        
        # Build top-down path
        for i in range(len(features) - 2, -1, -1):
            # Upsample current feature
            upsampled = self.upsample_convs[i](pyramid_features[-1])
            
            # Resize if shapes don't match
            if upsampled.shape[2:] != attended_laterals[i].shape[2:]:
                upsampled = F.interpolate(
                    upsampled, 
                    size=attended_laterals[i].shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Dynamic fusion gate
            concat = torch.cat([upsampled, attended_laterals[i]], dim=1)
            gates = self.fusion_gates[i](concat)
            
            # Apply gates
            fused = gates[:, 0:1, :, :] * upsampled + gates[:, 1:2, :, :] * attended_laterals[i]
            pyramid_features.append(fused)
        
        # Reverse to get from fine to coarse
        pyramid_features = pyramid_features[::-1]
        
        # Resize all features to target size and fuse
        resized_features = []
        for feature in pyramid_features:
            resized = F.interpolate(
                feature, 
                size=self.target_size, 
                mode='bilinear', 
                align_corners=False
            )
            resized_features.append(resized)
        
        # Combine all features (simple sum for now, can be enhanced)
        fused_feature = sum(resized_features)
        output = self.output_conv(fused_feature)
        
        # Convert to sequence format for decoder
        B, C, H, W = output.shape
        output = output.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        return output

class ConvNeXtMambaForCoCa(nn.Module):
    def __init__(
        self,
        image_size=224,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        d_state=16,
        d_conv=4,
        expand=2,
        drop_path_rate=0.,
        target_dim=512  # Final dimension for CoCa
    ):
        super().__init__()
        
        # Initial stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            DyT(dims[0], data_format="channels_first")  # Thay thế LayerNorm
        )

        # Drop path rate for each stage
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # ConvNeXt stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = ConvNeXtStage(
                dim_in=dims[i-1] if i > 0 else dims[0],
                dim_out=dims[i],
                depth=depths[i],
                drop_path=dp_rates[cur:cur+depths[i]][-1]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Mamba blocks - one for each stage
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                dim=dims[i],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for i in range(4)
        ])

        # Scale-Adaptive Pyramid Decoder
        self.pyramid_decoder = PyramidFeatureDecoder(
            dims=dims,
            target_dim=target_dim
        )

    def forward(self, x):
        x = self.stem(x)
        
        conv_outputs = []
        mamba_outputs = []
        
        # Process through stages
        for i, (stage, mamba) in enumerate(zip(self.stages, self.mamba_blocks)):
            # ConvNeXt processing
            x = stage(x)
            conv_outputs.append(x)
            
            # Mamba processing on ConvNeXt output
            mamba_out = mamba(x)  # Now returns 2D tensor
            mamba_outputs.append(mamba_out)
            
            # Next stage input comes from ConvNeXt output
            if i < len(self.stages) - 1:
                x = x  # Already have the ConvNeXt output
        
        # Process through Scale-Adaptive Pyramid Decoder
        final_features = self.pyramid_decoder(mamba_outputs)
        
        return final_features

def create_convnext_mamba_coca(
    image_size=224,
    in_chans=3,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    d_state=16,
    d_conv=4,
    expand=2,
    drop_path_rate=0.,
    target_dim=512,
    **kwargs
):
    model = ConvNeXtMambaForCoCa(
        image_size=image_size,
        in_chans=in_chans,
        depths=depths,
        dims=dims,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        drop_path_rate=drop_path_rate,
        target_dim=target_dim
    )
    return model

# Configuration examples
base_config = dict(
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    d_state=16,
    d_conv=4,
    expand=2,
    drop_path_rate=0.1,
    target_dim=512
)

small_config = dict(
    depths=[2, 2, 6, 2],
    dims=[96, 192, 384, 512],
    d_state=16,
    d_conv=4,
    expand=2,
    drop_path_rate=0.1,
    target_dim=512
) 