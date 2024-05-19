import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualTransformerBlock(nn.Module):
    def __init__(self, in_channels, nhead, num_layers):
        super(ResidualTransformerBlock, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=in_channels, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(in_channels)
        self.se = SEBlock(in_channels)

    def forward(self, src):
        # Reshape 4D tensor (B, C, H, W) to 3D tensor (B, C, H*W)
        B, C, H, W = src.shape
        src = src.view(B, C, H * W).permute(2, 0, 1)  # (H*W, B, C)
        src2 = self.transformer_encoder(src)
        src = src + src2  # Residual connection
        src = self.norm(src)
        src = self.se(src.permute(1, 2, 0).view(B, C, H, W))  # Reshape back to 4D tensor
        return src

class TransMUNet(nn.Module):
    def __init__(self, in_channels, out_channels, nhead, num_layers):
        super(TransMUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualTransformerBlock(64, nhead, num_layers),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)  # Ensure final output has correct dimensions
        )
        self.n_classes = out_channels

    def forward(self, x, istrain=True):
        x = self.encoder(x)
        x = self.decoder(x)
        return x