import torch
import torch.nn as nn
from .cbam import CBAM
from einops import rearrange
import torch.nn.functional as F
import numbers


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        # x = self.relu(x)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.relu = GELU()
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out = torch.cat((out1, out2), dim=1)
        return self.out_conv(self.relu(out))


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class SelfAtten(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.to_q = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)

        self.dw_q = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.dw_k = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.dw_v = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim, bias=False)
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1_spa = LayerNorm(in_dim)
        self.norm1_spe = LayerNorm(in_dim)
        self.norm2 = LayerNorm(in_dim)
        self.norm_out = LayerNorm(in_dim)
        self.proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, bias=False)
        self.ffn = PreNorm(in_dim)

    def forward(self, x):
        _, c, h, w = x.shape
        x_norm = self.norm1_spa(x)

        q = self.dw_q(self.to_q(x_norm))
        k = self.dw_k(self.to_k(x_norm))
        v = self.dw_v(self.to_v(x_norm))

        q = rearrange(q, 'b (n c) h w -> b n c (h w)', n=self.num_heads)
        k = rearrange(k, 'b (n c) h w -> b n c (h w)', n=self.num_heads)
        v = rearrange(v, 'b (n c) h w -> b n c (h w)', n=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b n c (h w) -> b (n c) h w', h=h, w=w)

        out = self.norm2(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, 1, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, bias=False, groups=dim * 4),
            nn.LeakyReLU(),
            nn.Conv2d(dim * 4, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.conv(x) + x


class StageBlock(nn.Module):
    def __init__(self, dim):
        super(StageBlock, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(dim, dim),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.trans = SelfAtten(dim, num_heads=dim // 4)
        self.ffn = FFN(dim)
        self.conv_out = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        f_conv = self.conv(x)
        f_trans = self.trans(x)
        fea = f_conv + f_trans + x
        out = self.ffn(fea) + fea
        return self.conv_out(out)


class PMM(nn.Module):
    def __init__(self, dim, stage_num=3):
        super(PMM, self).__init__()
        self.conv_init = nn.Sequential(
            ConvBNReLU(1, dim),
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        dim *= 2
        self.encoder = nn.ModuleList()
        for i in range(stage_num):
            self.encoder.append(
                nn.ModuleList([
                    StageBlock(dim),
                    nn.Sequential(
                        nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(inplace=True)
                    )
                ])
            )
            dim *= 2

        self.bottleneck = StageBlock(dim=dim)

        self.decoder = nn.ModuleList()
        for i in range(stage_num):
            self.decoder.append(
                nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(dim, dim * 2, 1),
                        nn.PixelShuffle(2)
                    ),
                    nn.Conv2d(dim, dim // 2, 1, 1),
                    StageBlock(dim // 2),
                ])
            )
            dim = dim // 2

        self.conv_last = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(dim // 2, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        f_x = self.conv_init(x)

        x = f_x

        fea_list = []
        for stage_block, Downsample in self.encoder:
            x = stage_block(x)
            fea_list.append(x)
            x = Downsample(x)
        x = self.bottleneck(x)

        for i, [Upsample, Fusion, stage_block] in enumerate(self.decoder):
            x = Upsample(x)
            x = Fusion(torch.cat([x, fea_list.pop()], dim=1))
            x = stage_block(x)
        out = self.conv_last(torch.cat([x, f_x], dim=1))
        return out


class DRM(nn.Module):
    def __init__(self, para_num, dim):
        super().__init__()
        self.init_conv = nn.Conv2d(para_num, 1, kernel_size=1, bias=False)
        self.forward_map = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.backward_map = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1)
        )

        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))
        self.phi = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim)
        self.gamma = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False, groups=dim)
        self.former_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.ffn = ConvBNReLU(dim, dim)

        self.cbam = CBAM(dim)

    def forward(self, paras, factor_pre=None):
        paras = torch.cat(paras, dim=1)
        radio_map_init = self.init_conv(paras)
        radio_map = self.forward_map(radio_map_init)
        factor_now = radio_map
        if factor_pre is not None:
            factor_pre = self.cbam(factor_pre)
            x_for = self.former_conv(factor_pre)
            phi = torch.sigmoid(self.phi(x_for))
            gamma = self.gamma(x_for)
            radio_map = phi * radio_map + gamma
            radio_map = self.ffn(radio_map)

        radio_map = torch.mul(torch.sign(radio_map), F.relu(torch.abs(radio_map) - self.soft_thr))
        radio_map = self.backward_map(radio_map)

        return radio_map, factor_now


class GDM(nn.Module):
    def __init__(self, dim, para_num):
        super().__init__()
        self.para_num = para_num
        self.lambda_steps = []
        for idx in range(para_num + 1):
            self.lambda_steps.append(nn.Parameter(torch.Tensor([0.5])).cuda())
            self.add_module(f'forward_p_{idx}', nn.Sequential(
                nn.Conv2d(1, dim, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ))
            self.add_module(f'backward_p_{idx}', nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(dim, 1, kernel_size=3, padding=1)
            ))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

    def forward(self, paras_sum, paras, sample_mask, PhiT, sample):
        h = paras_sum.shape[2]
        paras_renew = []
        for idx in range(self.para_num):
            para_sum = rearrange(paras_sum, 'b c h w -> b c (h w)')
            renew = para_sum @ sample_mask - sample
            renew_para = F.linear(renew, PhiT)
            renew_para = rearrange(renew_para, 'b c (h w) -> b c h w', h=h)
            renew_para = self.lambda_steps[idx] * renew_para
            para_new = paras[idx] + renew_para
            para_new = self.__getattr__(f'forward_p_{idx}')(para_new)
            para_new = torch.mul(torch.sign(para_new), F.relu(torch.abs(para_new) - self.soft_thr))
            para_new = self.__getattr__(f'backward_p_{idx}')(para_new)
            paras_renew.append(para_new)
            paras_sum = paras_sum + para_new
        return paras_renew


class UnfoldingBlock(nn.Module):
    def __init__(self, dim, para_num):
        super(UnfoldingBlock, self).__init__()

        self.para_num = para_num
        self.gdm = GDM(dim, para_num)
        self.drm = DRM(para_num, dim)
        self.pmm = PMM(dim)

    def forward(self, x, paras, sample_mask, PhiT, sample, factor_pre=None):
        paras_sum = x
        paras_renew = self.gdm(paras_sum, paras, sample_mask, PhiT, sample)
        radio_coarse, factor_now = self.drm(paras_renew, factor_pre)
        re_x = self.pmm(radio_coarse)
        return re_x, paras_renew, factor_now
