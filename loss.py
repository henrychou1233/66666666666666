#loss.py
import torch
import numpy as np
import logging
from typing import Optional
from feature_extractor import build_feature_extractor

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ----------------------- 頂部選項（在這裡選擇 noise_type） -----------------------
# 可選: "gaussian_opt" (clipped Gaussian) 或 "gaussian" (標準 Gaussian)
# 若 config.model.noise_type 有設定，會以 config 優先；否則使用這裡的 NOISE_TYPE。
NOISE_TYPE = "gaussian_opt"
# ------------------------------------------------------------------------------

# ---------------- 全域設定 ----------------
USE_FF_LOSS = True     # 是否使用 Feature Loss (多尺度引導)
RAW_WEIGHT = 1.0         # Raw Loss 權重
FEAT_WEIGHT = 1      # Feature Loss 權重
FEATURE_LAYER_WEIGHTS = [1, 1, 1]  # 每層 Feature 的相對權重

USE_PENALTY = False       # 是否加懲罰項
PENALTY_WEIGHT = 0.1   # 懲罰項的權重

_feature_extractor_model = None


# ---------------- Feature Extractor ----------------
def init_feature_extractor(config):
    global _feature_extractor_model
    if _feature_extractor_model is None:
        _feature_extractor_model = build_feature_extractor(config).to(config.model.device)
        
        # ✅ 強制凍結所有參數
        for p in _feature_extractor_model.parameters():
            p.requires_grad = False
        _feature_extractor_model.eval()

        # ✅ 印出確認訊息
        print(f"[Feature Extractor] Backbone: {_feature_extractor_model.__class__.__name__} | Status: Frozen ✅ (no grad update)")
    return _feature_extractor_model




# ---------------- Adaptive Smooth Bi ----------------
class AdaptiveSmoothBi:
    def __init__(self, l1_init=0.1, l2_init=0.9,
                 bi_patience=20000, bi_weight_step=0.005,
                 rel_improve_thresh=0.003, min_weight=0.01, max_weight=0.99):
        self.l1_weight = l1_init
        self.l2_weight = l2_init
        self.best_l1 = float("inf")
        self.best_l2 = float("inf")
        self.no_improve_count = 0
        self.bi_patience = bi_patience
        self.bi_weight_step = bi_weight_step
        self.rel_improve_thresh = rel_improve_thresh
        self.min_weight = min_weight
        self.max_weight = max_weight

    def step(self, base_loss, l1_err, l2_err):
        try:
            l1_val = float(torch.nan_to_num(l1_err, nan=1e3, posinf=1e3, neginf=-1e3))
            l2_val = float(torch.nan_to_num(l2_err, nan=1e3, posinf=1e3, neginf=-1e3))
        except Exception:
            l1_val, l2_val = 1e3, 1e3

        if l1_val < self.best_l1 * (1.0 - self.rel_improve_thresh):
            self.best_l1 = l1_val
            self.l1_weight = min(self.max_weight, self.l1_weight + self.bi_weight_step)
            self.l2_weight = max(self.min_weight, 1.0 - self.l1_weight)

        if l2_val < self.best_l2 * (1.0 - self.rel_improve_thresh):
            self.best_l2 = l2_val
            self.l2_weight = min(self.max_weight, self.l2_weight + self.bi_weight_step)
            self.l1_weight = max(self.min_weight, 1.0 - self.l2_weight)

        return torch.nan_to_num(base_loss, nan=1.0, posinf=1.0, neginf=-1.0)

    def get_weights(self):
        return self.l1_weight, self.l2_weight


# ---------------- Global Adaptive ----------------
_adaptive_raw = AdaptiveSmoothBi()
_feature_adaptives = [AdaptiveSmoothBi() for _ in range(len(FEATURE_LAYER_WEIGHTS))]


# ---------------- Noise Schedule (linear_opt) ----------------
def get_linear_opt_schedule(T, beta_start=1e-4, beta_end=0.02, alpha=1.4, offset=0.1,
                            beta_min=0.0, beta_max=0.2, smooth_kernel=3):
    x = np.linspace(0, 1, T)
    x = np.clip(x + offset, 0, 1)
    betas = beta_start + (beta_end - beta_start) * (x ** alpha)
    betas = np.clip(betas, beta_min, beta_max)

    if smooth_kernel > 1:
        pad = (smooth_kernel - 1) // 2
        padded = np.pad(betas, (pad, pad), mode="edge")
        kernel = np.ones(smooth_kernel) / smooth_kernel
        betas = np.convolve(padded, kernel, mode="valid")

    return betas.astype(np.float64)


# ---------------- Gaussian_opt Noise (只加 clip) ----------------
def sample_gaussian_opt(x, mean=0.0, std_scale=1.0, min_val=-3.0, max_val=3.0):
    """
    gaussian_opt: 與標準 Gaussian 等價，但在最後對樣本做 clamp 處理以去除極端值。
    """
    e = torch.randn_like(x) * std_scale + mean
    return torch.clamp(e, min_val, max_val)


# ---------------- Get Loss ----------------
def get_loss(model, x_0, t, config, optimizer: Optional[torch.optim.Optimizer] = None, mask=None):
    """
    訓練時的 loss 計算：
    - schedule: linear_opt (get_linear_opt_schedule)
    - noise: 可選 'gaussian_opt' 或 'gaussian'
      順序：若 config.model.noise_type 有設定 -> 以 config 為準；否則使用檔案頂部 NOISE_TYPE。
    """
    device = config.model.device
    x_0 = x_0.to(device)

    # ---------------- schedule (linear_opt) ----------------
    betas = get_linear_opt_schedule(
        config.model.trajectory_steps,
        beta_start=getattr(config.model, "beta_start", 1e-4),
        beta_end=getattr(config.model, "beta_end", 0.02),
        alpha=getattr(config.model, "linear_opt_alpha", 1.4),
        offset=getattr(config.model, "linear_opt_offset", 0.1),
        beta_min=getattr(config.model, "linear_opt_min", 0.0),
        beta_max=getattr(config.model, "linear_opt_max", 0.2),
        smooth_kernel=getattr(config.model, "schedule_smoothing_kernel", 3)
    )
    b = torch.tensor(betas, dtype=torch.float32, device=device)

    # ---------------- noise type 選項（config 優先，否則用檔案頂部 NOISE_TYPE） ----------------
    noise_type = getattr(config.model, "noise_type", NOISE_TYPE).lower()

    if noise_type == "gaussian_opt":
        # 只做 clip，其他與標準 Gaussian 相同
        mean = float(getattr(config.model, "gaussian_opt_mean", 0.0))
        std_scale = float(getattr(config.model, "gaussian_opt_std_scale", 1.0))
        min_val = float(getattr(config.model, "gaussian_opt_min", -3.0))
        max_val = float(getattr(config.model, "gaussian_opt_max", 3.0))
        e = sample_gaussian_opt(x_0, mean=mean, std_scale=std_scale, min_val=min_val, max_val=max_val)

    elif noise_type == "gaussian":
        e = torch.randn_like(x_0)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}' in config.model.noise_type or NOISE_TYPE. Choose 'gaussian_opt' or 'gaussian'.")

    # ---------------- forward noisy x_t ----------------
    at = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x_t = at.sqrt() * x_0 + (1 - at).sqrt() * e

    pred_noise = model(x_t, t.float())
    if mask is not None:
        e, pred_noise = e * mask, pred_noise * mask

    # --- RAW loss ---
    l1_err = (e - pred_noise).abs().sum(dim=(1, 2, 3)).mean()
    l2_err = (e - pred_noise).pow(2).sum(dim=(1, 2, 3)).mean()
    base_raw = _adaptive_raw.l1_weight * l1_err + _adaptive_raw.l2_weight * l2_err
    raw_loss = _adaptive_raw.step(base_raw, l1_err, l2_err)

    # --- reconstruct x0 ---
    x0_pred = (x_t - (1 - at).sqrt() * pred_noise) / at.sqrt()

    # --- Feature loss ---
    feat_losses = []
    if USE_FF_LOSS:
        fe = init_feature_extractor(config)
        with torch.no_grad():
            tgt_feats = fe(x_0)
        rec_feats = fe(x0_pred)

        for i, (tf, rf) in enumerate(zip(tgt_feats, rec_feats)):
            l1_feat = (rf - tf).abs().mean()
            l2_feat = (rf - tf).pow(2).mean()
            base_feat = _feature_adaptives[i].l1_weight * l1_feat + _feature_adaptives[i].l2_weight * l2_feat
            feat_loss = _feature_adaptives[i].step(base_feat, l1_feat, l2_feat)
            feat_losses.append(feat_loss)

    penalty = None
    if USE_PENALTY:
        penalty = torch.mean(pred_noise ** 2)

    total_loss = RAW_WEIGHT * raw_loss
    for i, feat_loss in enumerate(feat_losses):
        total_loss += FEAT_WEIGHT * FEATURE_LAYER_WEIGHTS[i] * feat_loss
    if USE_PENALTY and penalty is not None:
        total_loss += PENALTY_WEIGHT * penalty

    total_loss = torch.nan_to_num(total_loss, nan=1.0, posinf=1.0, neginf=-1.0)

    detail = {
        "raw": float(raw_loss.detach().cpu().item()),
        "features": [float(f.detach().cpu().item()) for f in feat_losses],
        "penalty": float(penalty.detach().cpu().item()) if penalty is not None else None,
        "combine": float(total_loss.detach().cpu().item())
    }

    return total_loss, detail


# ---------------- Reset state ----------------
def reset_adaptive_state():
    global _adaptive_raw, _feature_adaptives
    _adaptive_raw = AdaptiveSmoothBi()
    _feature_adaptives = [AdaptiveSmoothBi() for _ in range(len(FEATURE_LAYER_WEIGHTS))]


# ---------------- Print Weights ----------------
def print_adaptive_weights():
    raw_w = _adaptive_raw.get_weights()
    logging.info(f"[Epoch Adaptive] RAW L1_w={raw_w[0]:.4f}, L2_w={raw_w[1]:.4f}")

    for i, fa in enumerate(_feature_adaptives):
        w = fa.get_weights()
        logging.info(f"[Epoch Adaptive] FEAT[{i}] L1_w={w[0]:.4f}, L2_w={w[1]:.4f}")