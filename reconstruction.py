# ======================================================
# reconstruction_with_guidance.py — Dynamic RAW_W + EMA smoothing + log (2025-10-24)
# ======================================================
import os, math, logging
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# ---------------- GPU / 裝置設定 ----------------
CUDA_VISIBLE_DEVICES = "0,1,2"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Guidance 總體 & 獨立權重 ----------------
GUIDANCE_W: float = 1
RAW_W: float =3# Raw guidance 基準權重
FEAT_W: float =3# Multi-scale guidance
CLIP_W: float = 0      # CLIP guidance

# ---------------- Multi-scale guidance 預設參數 ----------------
MULTISCALE_GUIDANCE_ENABLED: bool = True
NORMALIZE_MS: bool = True
MULTISCALE_GUIDANCE_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "scale_weights": [1, 1, 1],
}

# ---------------- Semantic guidance ----------------
SEMANTIC_GUIDANCE_ENABLED: bool = True
NORMALIZE_CLIP: bool = True
CLIP_PROMPT: str = "a flawless can"
CLIP_CKPT_DIR: Path = Path("clipcheckpoints/potato/epoch_5").resolve()
CLIP_WEIGHTS: Dict[str, float] = {"cosine": 1.0, "dot": 0, "l2": 0, "l1": 0}
SEMANTIC_GUIDANCE_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "w": CLIP_W,
    "prompt": CLIP_PROMPT,
    "weights": CLIP_WEIGHTS
}

# ---------------- Noise / Schedule ----------------
DEFAULT_NOISE_TYPE: str = "gaussian_opt"
SCHEDULE_DEFAULTS: Dict[str, Any] = {
    "type": "linear_opt_v2",
    "trajectory_steps": 1000,
    "linear_opt_alpha": 1.4,
    "linear_opt_offset": 0.1,
    "linear_opt_min": 0.0,
    "linear_opt_max": 0.2,
    "linear_opt_beta_start": 1e-4,
    "linear_opt_beta_end": 0.02,
    "schedule_smoothing_kernel": 3,
}
GAUSSIAN_OPT_DEFAULTS: Dict[str, float] = {
    "mean": 0.0, "std_scale": 1.0, "min_val": -3.0, "max_val": 3.0
}

TEST_TRAJECTORY_STEPS_DEFAULT: int = 250
SKIP_DEFAULT: int = 25
NORMALIZE_EPS: float = 1e-12
SCHEDULE_CACHE_ENABLED: bool = True

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------- CLIP 初始化 ----------------
CLIP_MODEL: Optional[CLIPModel] = None
CLIP_PROCESSOR: Optional[CLIPProcessor] = None
if SEMANTIC_GUIDANCE_ENABLED:
    try:
        logging.info(f"[INFO] Loading CLIP model from: {CLIP_CKPT_DIR}")
        CLIP_MODEL = CLIPModel.from_pretrained(str(CLIP_CKPT_DIR), local_files_only=True).to(DEVICE)
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained(str(CLIP_CKPT_DIR), local_files_only=True)
        CLIP_MODEL.eval()
    except Exception as e:
        SEMANTIC_GUIDANCE_ENABLED = False
        logging.warning(f"[WARN] Failed to load CLIP model: {e}. Disabled.")

_to_pil = transforms.ToPILImage()
_SCHEDULE_CACHE: Dict[Tuple, np.ndarray] = {}

def _moving_average(arr: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return arr
    pad = (k - 1) // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode="valid")

def get_beta_schedule_cached(schedule_type: str, steps: int, schedule_params: Dict[str, Any]) -> np.ndarray:
    key = (schedule_type, int(steps))
    if SCHEDULE_CACHE_ENABLED and key in _SCHEDULE_CACHE:
        return _SCHEDULE_CACHE[key]
    beta_start = float(schedule_params.get("linear_opt_beta_start", 1e-4))
    beta_end = float(schedule_params.get("linear_opt_beta_end", 0.02))
    alpha = float(schedule_params.get("linear_opt_alpha", 1.3))
    offset = float(schedule_params.get("linear_opt_offset", 0.0))
    minv = float(schedule_params.get("linear_opt_min", 0.0))
    maxv = float(schedule_params.get("linear_opt_max", 0.1))
    if schedule_type in ("linear_opt", "linear_opt_v2"):
        x = np.linspace(0, 1, steps)
        x = np.clip(x + offset, 0.0, 1.0)
        betas = beta_start + (beta_end - beta_start) * (x ** alpha)
        betas = np.clip(betas, minv, maxv)
        if schedule_type == "linear_opt_v2":
            k = max(1, int(schedule_params.get("schedule_smoothing_kernel", 3)))
            betas = _moving_average(betas, k)
    elif schedule_type == "linear":
        betas = np.linspace(beta_start, beta_end, steps, dtype=np.float64)
    elif schedule_type == "cosine":
        s = 0.008
        t = np.linspace(0, steps, steps + 1) / steps + s
        alphas = np.cos(0.5 * np.pi * t / (1 + s)) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, 0, 0.999).astype(np.float64)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")
    betas = betas.astype(np.float64)
    if SCHEDULE_CACHE_ENABLED:
        _SCHEDULE_CACHE[key] = betas
    return betas

def clip_anomaly_scores_batch(images_tensor: torch.Tensor, prompt: str, weights: Dict[str, float]) -> torch.Tensor:
    if not SEMANTIC_GUIDANCE_ENABLED or CLIP_MODEL is None or CLIP_PROCESSOR is None:
        return torch.zeros(images_tensor.shape[0], 1, 1, 1, device=images_tensor.device)
    pil_images = []
    for k in range(images_tensor.shape[0]):
        img = (images_tensor[k].detach().cpu() + 1.0) / 2.0
        pil_images.append(_to_pil(img.clamp(0.0, 1.0)))
    inputs = CLIP_PROCESSOR(images=pil_images, text=[prompt] * len(pil_images), return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = CLIP_MODEL(**inputs)
    img_emb = outputs.image_embeds
    txt_emb = outputs.text_embeds
    img_norm = img_emb / (img_emb.norm(dim=-1, keepdim=True) + NORMALIZE_EPS)
    txt_norm = txt_emb / (txt_emb.norm(dim=-1, keepdim=True) + NORMALIZE_EPS)
    scores_list = []
    total_w = sum(weights.values()) if weights else 1.0
    for key, w in weights.items():
        if key == "cosine":
            sim = torch.sum(img_norm * txt_norm, dim=-1)
            val = (1.0 - sim) * float(w)
        elif key == "dot":
            sim = torch.sum(img_emb * txt_emb, dim=-1)
            val = (-sim) * float(w)
        elif key == "l2":
            val = torch.norm(img_emb - txt_emb, p=2, dim=-1) * float(w)
        elif key == "l1":
            val = torch.norm(img_emb - txt_emb, p=1, dim=-1) * float(w)
        else:
            continue
        scores_list.append(val)
    if len(scores_list) == 0:
        return torch.zeros(images_tensor.shape[0], 1, 1, 1, device=images_tensor.device)
    combined = sum(scores_list) / float(total_w)
    if NORMALIZE_CLIP:
        norm_val = combined.norm(p=2) + NORMALIZE_EPS
        combined = combined / norm_val
    return combined.view(-1, 1, 1, 1).to(images_tensor.device)

def multi_scale_guidance(yt, xt, feature_extractor, w: float = 1.0, scale_weights: Optional[List[float]] = None):
    feats_yt = feature_extractor(yt)
    feats_xt = feature_extractor(xt)
    if not isinstance(feats_yt, (list, tuple)):
        feats_yt, feats_xt = [feats_yt], [feats_xt]
    L = len(feats_yt)
    weights = scale_weights if (scale_weights is not None and len(scale_weights) == L) else [1.0 / L] * L
    device = yt.device
    multi_scale_diff = torch.zeros(yt.shape[0], 1, yt.shape[2], yt.shape[3], device=device, dtype=yt.dtype)
    for i, (f_y, f_x) in enumerate(zip(feats_yt, feats_xt)):
        diff = torch.mean((f_y - f_x) ** 2, dim=1, keepdim=True)
        diff_up = F.interpolate(diff, size=yt.shape[2:], mode='bilinear', align_corners=True)
        multi_scale_diff += weights[i] * diff_up
    return w * multi_scale_diff

# ======================================================
# Reconstruction 主類別
# ======================================================
class Reconstruction:
    def __init__(self, unet, config, noise_type: Optional[str] = None, schedule_type: Optional[str] = None, feature_extractor=None, debug: bool = False):
        self.unet = unet
        self.config = config
        self.feature_extractor = feature_extractor
        self.noise_type = noise_type if noise_type else DEFAULT_NOISE_TYPE
        self.schedule_type = schedule_type if schedule_type else SCHEDULE_DEFAULTS["type"]
        self.debug = debug
        self.ms_cfg = SimpleNamespace(**MULTISCALE_GUIDANCE_DEFAULTS)
        self.semantic_cfg = SimpleNamespace(**SEMANTIC_GUIDANCE_DEFAULTS)
        self._schedule_tensors: Dict[Tuple[str, int], Dict[str, torch.Tensor]] = {}
        self.prev_raw_w = RAW_W
        self.ema_alpha = 0.8
        try:
            self.unet.to(DEVICE).eval()
        except: pass
        if self.feature_extractor is not None:
            try: self.feature_extractor.to(DEVICE).eval()
            except: pass

    def _get_schedule_tensors(self, steps: int) -> Dict[str, torch.Tensor]:
        key = (self.schedule_type, int(steps))
        if key in self._schedule_tensors: return self._schedule_tensors[key]
        betas_np = get_beta_schedule_cached(self.schedule_type, steps, SCHEDULE_DEFAULTS)
        betas_t = torch.tensor(betas_np, dtype=torch.float32, device=DEVICE)
        b_padded = torch.cat([torch.zeros(1, device=DEVICE), betas_t], dim=0)
        at = (1.0 - b_padded).cumprod(dim=0)
        sqrt_at = at.sqrt()
        sqrt_1m_at = (1.0 - at).clamp(min=0.0).sqrt()
        tensors = {"betas": betas_t, "at": at, "sqrt_at": sqrt_at, "sqrt_1m_at": sqrt_1m_at}
        self._schedule_tensors[key] = tensors
        return tensors

    def add_noise(self, x, at, noise_type: Optional[str] = None):
        noise_type = noise_type if noise_type else self.noise_type
        device = x.device
        if noise_type == "gaussian_opt":
            mean, std = GAUSSIAN_OPT_DEFAULTS["mean"], GAUSSIAN_OPT_DEFAULTS["std_scale"]
            e = torch.randn_like(x, device=device) * std + mean
            e = torch.clamp(e, GAUSSIAN_OPT_DEFAULTS["min_val"], GAUSSIAN_OPT_DEFAULTS["max_val"])
        elif noise_type == "gaussian":
            e = torch.randn_like(x, device=device)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        if not isinstance(at, torch.Tensor):
            at = torch.tensor(at, dtype=x.dtype, device=device)
        sqrt_at = at.sqrt().view(-1, 1, 1, 1)
        sqrt_1_at = (1.0 - at).clamp(min=0.0).sqrt().view(-1, 1, 1, 1)
        return torch.nan_to_num(sqrt_at * x + sqrt_1_at * e, nan=0.0)

    # ======================================================
    # 🔹 Reconstruction Forward (with Dynamic RAW_W + EMA)
    # ======================================================
    def __call__(self, x, y0, w: float, test_trajectory_steps: Optional[int] = None, skip: Optional[int] = None, is_dynamic: bool = False):
        test_trajectory_steps = int(getattr(self.config.model, "test_trajectoy_steps", test_trajectory_steps or TEST_TRAJECTORY_STEPS_DEFAULT))
        skip = int(getattr(self.config.model, "skip", skip or SKIP_DEFAULT))
        sched = self._get_schedule_tensors(self.config.model.trajectory_steps)
        sqrt_at_full = sched["sqrt_at"]
        xt = self.add_noise(x.to(DEVICE), sqrt_at_full[test_trajectory_steps])
        seq = list(range(0, test_trajectory_steps, skip)) or [0]
        xs = [xt]; n = x.size(0)
        y0_device = y0.to(DEVICE)
        with torch.no_grad():
            seq_next = [-1] + list(seq[:-1])
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n, device=DEVICE) * i).long()
                next_t = (torch.ones(n, device=DEVICE) * j).long()
                at_t = sqrt_at_full.index_select(0, t + 1).view(-1, 1, 1, 1)
                at_next_t = sqrt_at_full.index_select(0, next_t + 1).view(-1, 1, 1, 1)
                xt = xs[-1]; et = self.unet(xt, t.float())
                yt = at_t.sqrt() * y0_device + (1 - at_t).sqrt() * et
                raw_guidance = yt - xt

                ms_guidance = 0.0
                if MULTISCALE_GUIDANCE_ENABLED and self.feature_extractor is not None:
                    ms_guidance = multi_scale_guidance(yt, xt, self.feature_extractor, w=1.0, scale_weights=self.ms_cfg.scale_weights)
                    if NORMALIZE_MS:
                        ms_guidance = ms_guidance / (ms_guidance.norm(p=2) + NORMALIZE_EPS)

                semantic_guidance = 0.0
                if SEMANTIC_GUIDANCE_ENABLED:
                    clip_scores = clip_anomaly_scores_batch(yt, self.semantic_cfg.prompt, self.semantic_cfg.weights)
                    semantic_guidance = -clip_scores.to(yt.device).type(yt.dtype)

                # ======================================================
                # 🔹 動態 RAW_W 調整 + EMA 平滑 + Log
                # ======================================================
                if isinstance(w, torch.Tensor):
                    dynamic_factor = w.mean().item()
                else:
                    dynamic_factor = float(w)
                dynamic_factor = max(0.9, min(1.1, dynamic_factor))
                target_raw_w = RAW_W * dynamic_factor
                prev_raw_w = self.prev_raw_w
                self.prev_raw_w = self.ema_alpha * self.prev_raw_w + (1 - self.ema_alpha) * target_raw_w
                dynamic_raw_w = self.prev_raw_w

                logging.info(f"[Dynamic RAW_W] {prev_raw_w:.3f} → {dynamic_raw_w:.3f} (factor={dynamic_factor:.3f})")

                combine_term = GUIDANCE_W * (
                    dynamic_raw_w * raw_guidance +
                    FEAT_W * ms_guidance +
                    CLIP_W * semantic_guidance
                )

                et_hat = et - (1 - at_t).sqrt() * combine_term
                x0_t = (xt - et_hat * (1 - at_t).sqrt()) / (at_t.sqrt() + NORMALIZE_EPS)
                eta = float(getattr(self.config.model, "eta", 1.0))
                safe_div = (1 - at_t).clamp(min=0.0)
                safe_denom = (1 - at_next_t).clamp(min=0.0)
                tmp_ratio = ((1 - at_t / (at_next_t + NORMALIZE_EPS)) * safe_denom / (safe_div + NORMALIZE_EPS)).clamp(min=0.0)
                c1 = (eta * tmp_ratio.sqrt()).clamp(min=0.0)
                c2 = (1 - at_next_t - c1 ** 2).clamp(min=0.0).sqrt()
                noise = torch.clamp(torch.randn_like(x0_t), -3.0, 3.0)
                xt_next = at_next_t.sqrt() * x0_t + c1 * noise + c2 * et_hat
                xs.append(xt_next)

        logging.info(f"[Guidance Weights] TOTAL_W={GUIDANCE_W:.3f} | RAW={self.prev_raw_w:.3f} | FEAT={FEAT_W:.3f} | CLIP={CLIP_W:.3f}")
        return xs
