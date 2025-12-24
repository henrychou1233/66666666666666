# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# ------------------ 官方模型權重 ------------------ #
model_urls = {
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# ------------------ 基本卷積 ------------------ #
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# ============================================================
# Attention Modules
# ============================================================
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = torch.cat([avg_out, max_out], dim=1)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(spatial_attn))
        return x * spatial_attn


class SE(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CoordAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x).permute(0, 1, 3, 2)
        x_w = self.pool_w(x)
        y = torch.cat([x_h, x_w], dim=3)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=3)
        x_h = self.sigmoid(self.conv_h(x_h.permute(0, 1, 3, 2)))
        x_w = self.sigmoid(self.conv_w(x_w))
        return identity * x_h * x_w


class HaloAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
    def forward(self, x):
        return self.conv(x)


class QKVAttention(nn.Module):
    def __init__(self, channels, n_heads=4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, 3, self.n_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.inter_channels = max(1, channels // 2)
        self.g = nn.Conv2d(channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        g_x = self.g(x).view(b, self.inter_channels, -1)
        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        f = torch.bmm(theta_x.transpose(1, 2), phi_x)
        f_div_C = torch.softmax(f, dim=-1)
        y = torch.bmm(f_div_C, g_x.transpose(1, 2)).transpose(1, 2).contiguous()
        y = y.view(b, self.inter_channels, h, w)
        return self.W(y) + x

# ============================================================
# LayerScale / GEGLU / GRN
# ============================================================
class LayerScale(nn.Module):
    def __init__(self, channels, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones((channels)), requires_grad=True)
    def forward(self, x):
        return self.gamma.view(1, -1, 1, 1) * x

class GEGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)
    def forward(self, x):
        x_proj, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(F.gelu(x_proj) * gate)

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x

# ============================================================
# MultiAttention Fusion (no DropPath)
# ============================================================
class MultiAttention(nn.Module):
    def __init__(self, channels,
                 w_cbam=1, w_se=1, w_coord=1, w_halo=1, w_qkv=1, w_nonlocal=1,
                 hidden_dim=1024):
        super().__init__()
        self.branch_modules, self.branch_weights = [], []
        if w_cbam > 0: self.branch_modules.append(CBAM(channels)); self.branch_weights.append(w_cbam)
        if w_se > 0: self.branch_modules.append(SE(channels)); self.branch_weights.append(w_se)
        if w_coord > 0: self.branch_modules.append(CoordAttention(channels)); self.branch_weights.append(w_coord)
        if w_halo > 0: self.branch_modules.append(HaloAttention(channels)); self.branch_weights.append(w_halo)
        if w_qkv > 0: self.branch_modules.append(QKVAttention(channels)); self.branch_weights.append(w_qkv)
        if w_nonlocal > 0: self.branch_modules.append(NonLocalBlock(channels)); self.branch_weights.append(w_nonlocal)

        self.branches = nn.ModuleList(self.branch_modules)
        self.alpha = nn.Parameter(torch.tensor(self.branch_weights, dtype=torch.float32), requires_grad=True)
        self.layerscale = LayerScale(channels)
        self.grn = GRN(channels)
        self.ffn = GEGLU(channels, hidden_dim)

    def forward(self, x):
        if len(self.branches) == 0:
            return x
        outs = [branch(x) for branch in self.branches]
        weights = torch.softmax(self.alpha, dim=0)
        attn_out = sum(w * o for w, o in zip(weights, outs))
        attn_out = self.layerscale(attn_out)
        attn_out = self.grn(attn_out)
        return x + attn_out

# ============================================================
# ResNet Backbone
# ============================================================
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 with_multiattn=False, attn_kwargs=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.with_multiattn = with_multiattn
        self.attn = MultiAttention(planes * self.expansion, **(attn_kwargs or {})) if with_multiattn else None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        if self.with_multiattn and self.attn is not None:
            out = out + self.attn(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, with_multiattn_layer4=False, attn_kwargs_layer4=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       with_multiattn=with_multiattn_layer4,
                                       attn_kwargs=attn_kwargs_layer4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

        # === 印出模型參數總量 ===
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        print("=" * 46)
        print(f"🧠 TOTAL SYSTEM PARAMETERS SUMMARY")
        print("=" * 46)
        print(f"Total params       : {total_params:,} ({total_params/1e6:.2f} M)")
        print(f"Trainable params   : {trainable_params:,} ({trainable_params/1e6:.2f} M)")
        print(f"Non-trainable params: {non_trainable_params:,} ({non_trainable_params/1e6:.2f} M)")
        print("=" * 46 + "\n")

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    with_multiattn=False, attn_kwargs=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        for i in range(blocks):
            use_multiattn = with_multiattn and (i == blocks - 1)
            layers.append(block(self.inplanes if i == 0 else planes * block.expansion,
                                planes, stride if i == 0 else 1,
                                downsample if i == 0 else None,
                                self.groups, self.base_width, previous_dilation,
                                norm_layer, with_multiattn=use_multiattn,
                                attn_kwargs=attn_kwargs if use_multiattn else None))
            downsample = None
            stride = 1
            previous_dilation = self.dilation
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3]

# ============================================================
# Wrapper
# ============================================================
def _resnet(arch, block, layers, pretrained, progress,
            with_multiattn_layer4=False, attn_kwargs_layer4=None, **kwargs):
    model = ResNet(block, layers, with_multiattn_layer4=with_multiattn_layer4,
                   attn_kwargs_layer4=attn_kwargs_layer4, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model

def wide_resnet101_2(pretrained=False, progress=True,
                     w_cbam=1, w_se=1, w_coord=1, w_halo=1, w_qkv=1, w_nonlocal=1,
                     hidden_dim=1024, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    attn_kwargs = dict(
        w_cbam=w_cbam, w_se=w_se, w_coord=w_coord,
        w_halo=w_halo, w_qkv=w_qkv, w_nonlocal=w_nonlocal,
        hidden_dim=hidden_dim
    )
    with_multiattn = any([w_cbam, w_se, w_coord, w_halo, w_qkv, w_nonlocal])
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress,
                   with_multiattn_layer4=with_multiattn,
                   attn_kwargs_layer4=attn_kwargs,
                   **kwargs)
