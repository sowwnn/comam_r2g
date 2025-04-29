import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
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

class ConvNeXtMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, drop_path=0.):
        super().__init__()
        
        # ConvNeXt block
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Mamba block
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
    def forward(self, x):
        input = x
        
        # ConvNeXt processing
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Prepare for Mamba
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)  # (B, H, W, C) -> (B, L, C)
        
        # Mamba processing
        x = self.mamba(x)
        
        # Reshape back
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        return input + self.drop_path(x)

class SelectiveFeatureAggregation(nn.Module):
    def __init__(self, dims):
        super().__init__()
        
        # Map all features to final dimension
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(dim, dims[-1], 1)
            for dim in dims
        ])
        
        # Selective attention for each level
        self.selective_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dims[-1], dims[-1], 1),
                nn.Sigmoid()
            ) for _ in range(len(dims))
        ])
        
    def forward(self, features):
        transformed_features = []
        
        # Transform and apply attention to each feature level
        for i, feat in enumerate(features):
            trans_feat = self.lateral_convs[i](feat)
            attention = self.selective_attention[i](trans_feat)
            transformed_features.append(trans_feat * attention)
        
        # Fuse features from top to bottom
        for i in range(len(transformed_features)-1, 0, -1):
            transformed_features[i-1] = transformed_features[i-1] + F.interpolate(
                transformed_features[i],
                size=transformed_features[i-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
        return transformed_features[0]

class ConvNeXtMambaForCoCa(nn.Module):
    def __init__(
        self, 
        image_size=224,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[128, 256, 512, 512],  # Final dim must match CoCa's image_dim
        d_state=16,
        d_conv=4,
        expand=2,
        drop_path_rate=0.,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.dims = dims
        self.return_embeddings = True
        
        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # ConvNeXt + Mamba stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtMambaBlock(
                    dim=dims[i],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    drop_path=dp_rates[cur + j]
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Feature aggregation
        self.feature_aggregation = SelectiveFeatureAggregation(dims)
        
        # Final normalization
        self.final_norm = nn.LayerNorm(dims[-1])
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        feature_maps = []
        
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_maps.append(x)
            
        return feature_maps

    def forward(self, x):
        # Extract hierarchical features
        feature_maps = self.forward_features(x)
        
        # Aggregate features
        features = self.feature_aggregation(feature_maps)  # [B, C, H, W]
        
        # Convert to sequence format for CoCa
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
        features = self.final_norm(features)
        
        return features

def create_convnext_mamba_coca(
    image_size=224,
    pretrained=False,
    **kwargs
):
    model = ConvNeXtMambaForCoCa(
        image_size=image_size,
        **kwargs
    )
    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet")
    return model

# Configuration examples
base_config = dict(
    depths=[3, 3, 9, 3],
    dims=[128, 256, 512, 512],  # Final dim matches CoCa's image_dim
    d_state=16,
    d_conv=4,
    expand=2,
    drop_path_rate=0.1
)

small_config = dict(
    depths=[2, 2, 6, 2],
    dims=[96, 192, 384, 512],
    d_state=16,
    d_conv=4,
    expand=2,
    drop_path_rate=0.1
)