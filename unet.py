# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import math
from abc import abstractmethod
# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import math
from abc import abstractmethod
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


import math
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from typing import Iterable
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class PositionalEmbedding(nn.Module):
    # PositionalEmbedding
    """
    Computes Positional Embedding of the timestep
    """

    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# 改良版 Downsample / Upsample 模組
# ======================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# 🔹 通用激活函數選擇器
# ======================================================
def get_activation(act_type: str = "gelu"):
    """
    回傳指定的激活函數。
    支援類型：
      - "gelu"：高斯誤差線性單元（推薦，平滑且穩定）
      - "silu"：Sigmoid-Weighted Linear Unit（Swish）
      - "mish"：非線性較強，適合生成式任務
      - "relu"：經典 ReLU
      - "leakyrelu"：容忍負值輸入
    """
    act_type = act_type.lower()
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "silu":
        return nn.SiLU()
    elif act_type == "mish":
        return nn.Mish()
    elif act_type == "relu":
        return nn.ReLU(inplace=True)
    elif act_type == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")


# ======================================================
# 🔽 Downsample：下採樣模組（帶殘差旁路 + 自動對齊）
# ======================================================

# ======================================================
# 🔽 Downsample（使用 Mish）
# ======================================================
class Downsample(nn.Module):
    """
    改良版下採樣模組：
      - 支援卷積式或平均池化下採樣。
      - 使用 GroupNorm + Mish + Dropout。
      - 自動通道調整與尺寸對齊。
    """
    def __init__(self, in_channels, use_conv=True, out_channels=None,
                 advanced=True, dropout=0.2):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        self.advanced = advanced
        self.use_conv = use_conv

        if use_conv:
            self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.down = nn.AvgPool2d(kernel_size=2, stride=2)

        if advanced:
            num_groups = min(32, out_channels // 4) or 1
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self.act = Mish()   # ✅ 改回 Mish
            self.drop = nn.Dropout2d(dropout)

        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=2) if in_channels != out_channels else None

    def forward(self, x):
        assert x.shape[1] == self.channels

        h = self.down(x)
        if self.advanced:
            h = self.norm(h)
            h = self.act(h)
            h = self.drop(h)

        if self.residual is not None:
            res = self.residual(x)
            if res.shape[-1] != h.shape[-1] or res.shape[-2] != h.shape[-2]:
                res = F.interpolate(res, size=h.shape[-2:], mode='bilinear', align_corners=False)
            h = h + res

        return h


# ======================================================
# 🔼 Upsample（使用 Mish）
# ======================================================
class Upsample(nn.Module):
    """
    改良版上採樣模組：
      - 使用雙線性插值上採樣。
      - 使用 GroupNorm + Mish + Dropout。
      - 含殘差旁路，支援通道匹配。
    """
    def __init__(self, in_channels, use_conv=True, out_channels=None,
                 advanced=True, dropout=0.2):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        self.advanced = advanced
        out_channels = out_channels or in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1) if use_conv else None

        if advanced:
            num_groups = min(32, out_channels // 4) or 1
            self.norm = nn.GroupNorm(num_groups, out_channels)
            self.act = Mish()   # ✅ 改回 Mish
            self.drop = nn.Dropout2d(dropout)

        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        assert x.shape[1] == self.channels

        h = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if self.conv:
            h = self.conv(h)
        if self.advanced:
            h = self.norm(h)
            h = self.act(h)
            h = self.drop(h)

        if self.residual is not None:
            res = self.residual(x)
            res = F.interpolate(res, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + res

        return h

# ======================================================
# ✅ 使用範例
# ======================================================
"""
down = Downsample(64, use_conv=True, out_channels=128, advanced=True, dropout=0.1, act_type="gelu")
up   = Upsample(128, use_conv=True, out_channels=64, advanced=True, dropout=0.1, act_type="gelu")

x = torch.randn(1, 64, 256, 256)
y = down(x)
z = up(y)
print("Down:", y.shape, " Up:", z.shape)
"""




"""
# 統一開關，直接在外層定義即可
ADVANCED = True
DROPOUT_RATE = 0.1

down = Downsample(64, use_conv=True, out_channels=128, advanced=ADVANCED, dropout=DROPOUT_RATE)
up   = Upsample(128, use_conv=True, out_channels=64, advanced=ADVANCED, dropout=DROPOUT_RATE)
"""


# ----------------------
# Attention components
# ----------------------
class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads


    def forward(self, qkv, time=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0, \
            f"qkv width {width} not divisible by 3*{self.n_heads}"
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class ChannelAttention1d(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_ = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        attn = self.sigmoid(avg + max_)
        return x * attn


class SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg, max_], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class CBAM1d(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention1d(in_channels, reduction)
        self.spatial_attn = SpatialAttention1d(spatial_kernel)


    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class CoordAttention1d(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        mid = max(8, in_channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(mid, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        y = self.pool(x)
        y = self.act(self.bn1(self.conv1(y)))
        y = self.sigmoid(self.conv2(y))
        return x * y


class HaloAttention1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)


    def forward(self, x):
        return self.conv(x)


class NonLocal1d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2 if in_channels > 1 else 1
        self.g = nn.Conv1d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv1d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv1d(in_channels, self.inter_channels, 1)
        self.W = nn.Conv1d(self.inter_channels, in_channels, 1)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)


    def forward(self, x):
        # x: [B, C, L]
        batch_size, C, L = x.shape
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [B, C', L]
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # [B, C', L]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C', L]
        f = torch.matmul(theta_x.transpose(1, 2), phi_x)  # [B, L, L]
        f_div_C = torch.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x.transpose(1, 2))  # [B, L, C']
        y = y.transpose(1, 2).contiguous()
        W_y = self.W(y)
        z = W_y + x
        return z


class SE1d(nn.Module):
    """Squeeze-and-Excitation for 1D feature maps"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, C, L)
        b, c, l = x.shape
        y = self.avg_pool(x).view(b, c)      # (B, C)
        y = self.fc(y).view(b, c, 1)         # (B, C, 1)
        return x * y                         # scale each channel


import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

import torch.nn.functional as F
import torch
import torch.nn as nn
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, numpy as np

import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# 🔹 Attention Block（移除 FFN，保留多分支注意力）
# ======================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# ✅ Mish Activation Function
# ======================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# ======================================================
# 🔹 LocalEnhanceBlock：局部特徵強化模組（新增）
# ======================================================
class LocalEnhanceBlock(nn.Module):
    """用可分離卷積捕捉局部紋理特徵，加強區域細節與梯度穩定性"""
    def __init__(self, dim, expansion=2, kernel_size=3):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.conv_dw = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=False)
        self.conv_pw1 = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.act = nn.GELU()
        self.conv_pw2 = nn.Conv1d(hidden_dim, dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 平衡係數（局部與原特徵融合）

    def forward(self, x):
        residual = x
        out = self.conv_dw(x)
        out = self.act(self.conv_pw1(out))
        out = self.conv_pw2(out)
        out = self.bn(out)
        return residual * (1 - self.alpha) + out * self.alpha


# ======================================================
# 🔹 Attention Block（多分支注意力 + FFN + LocalEnhance）
# ======================================================
# ======================================================
# 🔹 Attention Block（固定權重版，多分支注意力 + FFN + LocalEnhance）
# ======================================================
class AttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        n_heads=1,
        n_head_channels=-1,
        use_ffn=True,           # ✅ 預設開啟 FFN
        ffn_ratio=4.0,
        ffn_dropout=0.1,
        use_local=True          # ✅ 預設開啟 LocalEnhanceBlock
    ):
        super().__init__()
        self.use_ffn = use_ffn
        self.use_local = use_local
        self.norm = GroupNorm32(32, in_channels)

        # ==========================
        # 🔸 多分支注意力模組
        # ==========================
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert in_channels % n_head_channels == 0
            self.num_heads = in_channels // n_head_channels

        # ✅ 所有注意力分支（固定啟用，不學權重）
        self.qkv_attn = QKVAttention(self.num_heads)
        self.to_qkv   = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.cbam     = CBAM1d(in_channels)
        self.coord    = CoordAttention1d(in_channels)
        self.halo     = HaloAttention1d(in_channels)
        self.nonlocal_attn = NonLocal1d(in_channels)
        self.se       = SE1d(in_channels)

        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))

        # ✅ 固定權重（不學習、不動態）
        self.fixed_weights = torch.tensor(
            [1, 1, 1, 1, 1, 1], dtype=torch.float32
        ) / 6.0  # 平均加權

        # ==========================
        # 🔸 FFN 模組
        # ==========================
        if self.use_ffn:
            hidden_dim = int(in_channels * ffn_ratio)
            self.ffn = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, 1),
                Mish(),
                nn.Dropout(ffn_dropout),
                nn.Conv1d(hidden_dim, in_channels, 1),
                nn.Dropout(ffn_dropout),
            )
            self.ffn_norm = GroupNorm32(32, in_channels)
            self.ffn_gamma = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))

        # ==========================
        # 🔸 LocalEnhanceBlock 模組
        # ==========================
        if self.use_local:
            self.local_enhance = LocalEnhanceBlock(in_channels, expansion=2)

    # ======================================================
    # Forward
    # ======================================================
    def forward(self, x, time=None):
        b, c, *spatial = x.shape
        length = int(torch.tensor(spatial).prod()) if spatial else 1
        x_flat = x.reshape(b, c, length)
        x_norm = self.norm(x_flat)

        # 🔸 各分支注意力
        attn_list = [
            self.qkv_attn(self.to_qkv(x_norm)),
            self.cbam(x_flat),
            self.coord(x_flat),
            self.halo(x_flat),
            self.nonlocal_attn(x_flat),
            self.se(x_flat),
        ]

        # 🔸 使用固定平均權重融合
        weights = self.fixed_weights.to(x.device)
        attn_sum = sum(w * attn for w, attn in zip(weights, attn_list))
        attn_sum = self.proj_out(attn_sum)

        # 🔸 殘差連接 (Attention)
        out = x_flat + self.gamma * attn_sum

        # 🔸 局部強化
        if self.use_local:
            out = self.local_enhance(out)

        # 🔸 FFN
        if self.use_ffn:
            ffn_in = self.ffn_norm(out)
            ffn_out = self.ffn(ffn_in)
            out = out + self.ffn_gamma * ffn_out

        return out.reshape(b, c, *spatial)












import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================================
# 🔹 E-SiLU (Enhanced SiLU)
# ======================================================
# ======================================================
# ✅ Enhanced SiLU (E-SiLU, Final Unified Version)
# ======================================================
import torch
import torch.nn as nn

class ESiLU(nn.Module):
    """
    E-SiLU: 強化版可學 SiLU
    - alpha: 初始非線性斜率 (預設 1.5)
    - learnable: 是否讓模型學習 alpha（True 時自動學習）
    """
    def __init__(self, alpha=1.5, learnable=True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        return x * torch.sigmoid(self.alpha * x)




# ======================================================
# 🔷 ResBlock（E-SiLU + Fixed Residual Balance）
# ======================================================
class ResBlock(TimestepBlock):
    """
    改良版殘差模組：
      - 使用 E-SiLU 激活（更強非線性與穩定性）
      - 支援上下採樣（Upsample/Downsample）
      - Zero-module 初始化確保 early stage 穩定
      - 自動對齊 skip connection 尺寸（避免 off-by-one）
    """
    def __init__(
        self,
        in_channels,
        time_embed_dim,
        dropout,
        out_channels=None,
        use_conv=False,
        up=False,
        down=False
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        # === 主分支輸入層 ===
        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            ESiLU(alpha=1.5),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # === 上下採樣設定 ===
        self.updown = up or down
        if up:
            self.h_upd = Upsample(in_channels, False)
            self.x_upd = Upsample(in_channels, False)
        elif down:
            self.h_upd = Downsample(in_channels, False)
            self.x_upd = Downsample(in_channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # === 時間嵌入層 ===
        self.embed_layers = nn.Sequential(
            ESiLU(alpha=1.5),
            nn.Linear(time_embed_dim, out_channels)
        )

        # === 輸出層 ===
        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            ESiLU(alpha=1.5),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        )

        # === Skip Connection ===
        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    # ======================================================
    # Forward 流程
    # ======================================================
    def forward(self, x, time_embed):
        # --- 上/下採樣 ---
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # --- 時間嵌入 ---
        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out

        # --- 輸出層 ---
        h = self.out_layers(h)

        # --- 自動尺寸對齊 ---
        if h.shape[-1] != x.shape[-1] or h.shape[-2] != x.shape[-2]:
            x = F.interpolate(x, size=h.shape[-2:], mode="bilinear", align_corners=False)

        # --- 殘差輸出 ---
        return self.skip_connection(x) + h







import torch
import torch.nn as nn
import torch.nn.functional as F




# ==========================
# ✅ UNetModel (E-SiLU)
# ==========================
class UNetModel(nn.Module):
    # UNet model
    def __init__(
            self,
            img_size,
            base_channels,
            conv_resample=True,
            n_heads=1,
            n_head_channels=-1,
            channel_mults="",
            num_res_blocks=2,
            dropout=0,
            attention_resolutions="32,16,8",
            biggan_updown=True,
            in_channels=1
            ):
        self.dtype = torch.float32
        super().__init__()

        if channel_mults == "":
            if img_size == 512:
                channel_mults = (0.5, 1, 1, 2, 2, 4, 4)
            elif img_size == 256:
                channel_mults = (1, 1, 2, 2, 4, 4)
            elif img_size == 128:
                channel_mults = (1, 1, 2, 3, 4)
            elif img_size == 64:
                channel_mults = (1, 2, 3, 4)
            elif img_size == 32:
                channel_mults = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {img_size}")

        attention_ds = [img_size // int(res) for res in attention_resolutions.split(",")]

        self.image_size = img_size
        self.in_channels = in_channels
        self.model_channels = base_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample

        self.dtype = torch.float32
        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        # ---------------- Time Embedding ----------------
        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
                PositionalEmbedding(base_channels, 1),
                nn.Linear(base_channels, time_embed_dim),
                ESiLU(),   # ✅ 改成 E-SiLU
                nn.Linear(time_embed_dim, time_embed_dim),
                )

        # 為時間嵌入添加梯度裁剪
        for param in self.time_embedding.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -1.0, 1.0))

        # ---------------- Down Path ----------------
        ch = int(channel_mults[0] * base_channels)
        self.down = nn.ModuleList(
                [TimestepEmbedSequential(nn.Conv2d(self.in_channels, base_channels, 3, padding=1))]
                )
        channels = [ch]
        ds = 1
        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        out_channels=base_channels * mult,
                        dropout=dropout,
                        )]
                ch = base_channels * mult

                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels,
                                    )
                            )
                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)
            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(
                        TimestepEmbedSequential(
                                ResBlock(
                                        ch,
                                        time_embed_dim=time_embed_dim,
                                        out_channels=out_channels,
                                        dropout=dropout,
                                        down=True
                                        )
                                if biggan_updown
                                else
                                Downsample(ch, conv_resample, out_channels=out_channels)
                                )
                        )
                ds *= 2
                ch = out_channels
                channels.append(ch)

        # ---------------- Middle Path ----------------
        self.middle = TimestepEmbedSequential(
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout
                        ),
                AttentionBlock(
                        ch,
                        n_heads=n_heads,
                        n_head_channels=n_head_channels
                        ),
                ResBlock(
                        ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout
                        )
                )

        # ---------------- Up Path ----------------
        self.up = nn.ModuleList([])
        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [
                    ResBlock(
                            ch + inp_chs,
                            time_embed_dim=time_embed_dim,
                            out_channels=base_channels * mult,
                            dropout=dropout
                            )
                    ]
                ch = base_channels * mult
                if ds in attention_ds:
                    layers.append(
                            AttentionBlock(
                                    ch,
                                    n_heads=n_heads,
                                    n_head_channels=n_head_channels
                                    ),
                            )

                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(
                            ResBlock(
                                    ch,
                                    time_embed_dim=time_embed_dim,
                                    out_channels=out_channels,
                                    dropout=dropout,
                                    up=True
                                    )
                            if biggan_updown
                            else
                            Upsample(ch, conv_resample, out_channels=out_channels)
                            )
                    ds //= 2
                self.up.append(TimestepEmbedSequential(*layers))

        # ---------------- Output ----------------
        self.out = nn.Sequential(
                GroupNorm32(32, ch),
                ESiLU(),   # ✅ 改成 E-SiLU
                zero_module(nn.Conv2d(base_channels * channel_mults[0], self.out_channels, 3, padding=1))
                )
        # === 印出模型參數總量 ===
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        print("=" * 50)
        print(f"🧠 TOTAL UNET PARAMETERS SUMMARY")
        print("=" * 50)
        print(f"Total params       : {total_params:,} ({total_params/1e6:.2f} M)")
        print(f"Trainable params   : {trainable_params:,} ({trainable_params/1e6:.2f} M)")
        print(f"Non-trainable params: {non_trainable_params:,} ({non_trainable_params/1e6:.2f} M)")
        print("=" * 50 + "\n")

    # ---------------- Forward ----------------
    def forward(self, x, time):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            time_embed = self.time_embedding(time)
            skips = []

            h = x.type(self.dtype)
            for i, module in enumerate(self.down):
                h = module(h, time_embed)
                if self.training:
                    h = torch.utils.checkpoint.checkpoint(lambda x: x, h)
                skips.append(h)
            
            h = self.middle(h, time_embed)
            
            for i, module in enumerate(self.up):
                skip = skips.pop()
                alpha = 0.5
                h = torch.cat([h * alpha, skip * (1 - alpha)], dim=1)
                h = module(h, time_embed)
                if self.training:
                    h = torch.utils.checkpoint.checkpoint(lambda x: x, h)
            
            h = h.type(x.dtype)
            h = self.out(h)
            if self.training:
                h = torch.clamp(h, -10.0, 10.0)
            return h



class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels):
        # 🔹 自動調整群組數，確保能整除
        num_groups = max(1, min(num_groups, num_channels))
        while num_channels % num_groups != 0:
            num_groups -= 1
        super().__init__(num_groups, num_channels)

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)



def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)