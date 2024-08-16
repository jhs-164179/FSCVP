import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, upsampling=False):
        super(PatchEmbed, self).__init__()
        if upsampling:
            self.proj = nn.ConvTranspose2d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.upsampling = upsampling
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.contiguous().view(B*T, C, H, W)
        # x = x.reshape(B*T, C, H, W)
        x = self.proj(x)
        x = self.norm(x)
        if self.upsampling:
            x = x.view(B, T, self.embed_dim, H*self.patch_size, W*self.patch_size)
        else:
            x = x.view(B, T, self.embed_dim, H//self.patch_size, W//self.patch_size)
        return x
    

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Conv2Former-based
class ConvMod(nn.Module):
    def __init__(self, dim, kernel_size, drop_path=.1, init_value=1e-2):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding='same', groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        a = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.a(self.norm1(x)))
        x = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.v(self.norm2(x)))
        x = a * x
        x = self.proj(x)

        return x


class EncoderLayer_S(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=.1, init_value=1e-2):
        super(EncoderLayer_S, self).__init__()
        self.attn = ConvMod(dim, kernel_size=kernel_size, drop_path=drop_path, init_value=init_value)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        a = self.attn(x)
        x = x + a
        x = self.norm(x)
        x = x.view(B, T, C, H, W)
        return x, a


# LSB-based
class FeedForward(nn.Module): 
    def __init__(self, dim, dilation=2, ratio=2, drop_path=.1, init_value=1e-2):
        super().__init__()
        # self.ff1 = nn.Sequential(
        #     nn.Conv3d(dim, ratio * dim, 3, 1, 1, bias=False),
        #     nn.GroupNorm(1, ratio * dim),
        #     nn.SiLU(inplace=True),
        # )
        # self.ff2 = nn.Sequential(
        #     nn.Conv3d(ratio * dim, dim, 3, 1, 1, bias=False),
        #     nn.GroupNorm(1, dim),
        #     nn.SiLU(inplace=True),
        # )
        self.ff1 = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            #     nn.GroupNorm(1, dim),
            nn.Conv3d(dim, dim, kernel_size=3, bias=False, padding='same', groups=dim, dilation=(dilation, 1, 1)),
            nn.GELU(),
            # nn.GroupNorm(1, ratio * dim),
            # nn.SiLU(inplace=True),

            # nn.GELU(),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'layer_scale_1', 'layer_scale_2'}
    
    def forward(self, x):
        # x = self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.ff1(x.permute(0, 2, 1, 3, 4)))
        # x = x.permute(0, 2, 1, 3, 4)
        # x = self.ff2(self.ff1(x.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        x = self.ff1(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return x


class EncoderLayer_T(nn.Module):
    def __init__(self, dim, ratio=2, kernel_size=7, dilation=2, drop_path=.1, init_value=1e-2): # dim --> seq_len * dim
        super(EncoderLayer_T, self).__init__()
        # self.attn = STBlock(dim, ratio=ratio, kernel_size=kernel_size)
        self.attn = FeedForward(dim, ratio=ratio, drop_path=drop_path, init_value=init_value, dilation=dilation)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        a = self.attn(x)
        x = x + a
        x = self.norm(x.contiguous().view(-1, C, H, W))
        x = x.view(B, T, C, H, W)
        return x, a


class DecoderLayer_S(nn.Module):
    def __init__(self, dim, ratio=2):
        super(DecoderLayer_S, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim*ratio, 1),

            nn.GELU(),
            nn.Conv2d(dim*ratio, dim, 1),
        )
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, enc_attn):
        B, T, C, H, W = x.shape
        x = x.contiguous().view(-1, C, H, W)
        # a = enc_attn
        # for ablation study
        # x = x + enc_attn
        x = self.mlp(x)
        x = self.norm(x)
        x = x.contiguous().view(B, T, C, H, W)
        return x


class DecoderLayer_T(nn.Module):
    def __init__(self, dim, ratio=2): # dim --> seq_len * dim
        super(DecoderLayer_T, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, dim*ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(dim*ratio, dim, 3, 1, 1),
        )
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, enc_attn):
        B, T, C, H, W = x.shape
        # for ablation study
        # x = x + enc_attn
        x = self.mlp(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = self.norm(x.contiguous().view(-1, C, H, W))
        x = x.view(B, T, C, H, W)
        return x
    
