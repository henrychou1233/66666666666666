# anomaly_map.py
# ==============================================================+
# 熱圖計算（含可選 FID 輸出 + 舊版對齊模式 + CLIPAD 輔助） v2025-09-25 (改良版 FID)
# ==============================================================+

import math
from typing import Optional, Tuple, Union, Dict
from pathlib import Path
import logging

import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
from kornia.filters import gaussian_blur2d
from kornia.metrics import ssim as kornia_ssim          # pip install kornia
from torchmetrics.image.fid import FrechetInceptionDistance  # pip install torchmetrics
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ────────【可調參數區】──────────────────────────
W_FD     = 1 # Feature distance
W_L1     = 6# Pixel L1 distance
W_L2     = 1# L2 distance
W_SSIM   = 0.1# SSIM difference
W_FID    = 0 # FID score
W_CLIPAD = 0   # CLIP anomaly

GAUSS_SIG_DEFAULT = 4
EPS      = 1e-8


def feature_distance(
    output: torch.Tensor,
    target: torch.Tensor,
    FE: torch.nn.Module,
    config,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    device = torch.device(getattr(config.model, "device", "cpu"))
    FE.to(device)
    FE.eval()

    if normalize:
        transform = _make_normalize_transform(device)
        output_in, target_in = transform(output), transform(target)
    else:
        output_in, target_in = (output + 1) / 2, (target + 1) / 2

    with torch.no_grad():
        feats_out, feats_tgt = FE(output_in), FE(target_in)

    if isinstance(feats_out, torch.Tensor): feats_out = [feats_out]
    if isinstance(feats_tgt, torch.Tensor): feats_tgt = [feats_tgt]

    out_size = int(getattr(config.data, "image_size", output.shape[-1]))
    B = output.shape[0]
    anomaly_map = torch.zeros((B, 1, out_size, out_size), device=device, dtype=feats_out[0].dtype)

    LAYER_WEIGHTS = [1,1,1]

    for i, (f_o, f_t) in enumerate(zip(feats_out, feats_tgt)):
        if f_o is None or f_t is None:
            continue

        p_o, p_t = patchify(f_o), patchify(f_t)
        cos = F.cosine_similarity(p_t, p_o, dim=1, eps=1e-6)
        diff_map = (1.0 - cos).unsqueeze(1)
        diff_map = F.interpolate(diff_map, size=(out_size, out_size), mode='bilinear', align_corners=True)
        w = LAYER_WEIGHTS[min(i, len(LAYER_WEIGHTS) - 1)]
        anomaly_map += w * diff_map

    anomaly_map = anomaly_map / (sum(LAYER_WEIGHTS[:len(feats_out)]) + 1e-8)
    return anomaly_map

# CLIPAD 相關參數
CLIP_PROMPT: str = "a flawless bagel"
CLIP_WEIGHTS: Dict[str, float] = {"cosine": 1.0, "dot": 0, "l2": 0, "l1": 0}
CLIP_CKPT_DIR: Path = Path("clipcheckpoints/bagel/epoch_5").resolve()
CLIP_LOCAL_FILES_ONLY: bool = True
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

_TENSOR_TO_PIL = transforms.ToPILImage()
CLIP_MODEL: Optional[CLIPModel] = None
CLIP_PROCESSOR: Optional[CLIPProcessor] = None
_CLIP_ENABLED: bool = False
# ───────────────────────────────────────────────────────────────


# ====== 初始化 CLIP ======
def _init_clip():
    global CLIP_MODEL, CLIP_PROCESSOR, _CLIP_ENABLED
    try:
        logging.info(f"[anomaly_utils] 嘗試載入 CLIP 權重: {CLIP_CKPT_DIR}")
        CLIP_MODEL = CLIPModel.from_pretrained(
            str(CLIP_CKPT_DIR), local_files_only=CLIP_LOCAL_FILES_ONLY
        ).to(DEVICE)
        CLIP_PROCESSOR = CLIPProcessor.from_pretrained(
            str(CLIP_CKPT_DIR), local_files_only=CLIP_LOCAL_FILES_ONLY
        )
        CLIP_MODEL.eval()
        _CLIP_ENABLED = True
        logging.info("[anomaly_utils] 成功載入 CLIP 權重。")
    except Exception as e_local:
        logging.warning(f"[anomaly_utils] CLIP 載入失敗: {e_local}")
        _CLIP_ENABLED = False
        CLIP_MODEL, CLIP_PROCESSOR = None, None

_init_clip()


# ====== Normalization for FE ======
def _make_normalize_transform(device: torch.device):
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32)
    def _transform(x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        mean_b = mean.view(1, -1, 1, 1)
        std_b  = std.view(1, -1, 1, 1)
        return (x - mean_b) / (std_b + 1e-12)
    return _transform


def _compute_kernel_size_for_sigma(sigma: float) -> int:
    k = max(3, 2 * int(4 * sigma + 0.5) + 1)
    if k % 2 == 0:
        k += 1
    return k


# ====== CLIP 異常分數 (batch) ======
def _clip_batch_scores(image_tensors: torch.Tensor, prompt: str = CLIP_PROMPT, weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    if not _CLIP_ENABLED or CLIP_MODEL is None or CLIP_PROCESSOR is None:
        return torch.zeros((image_tensors.size(0),), device=image_tensors.device)
    if weights is None: weights = {"cosine": 1.0}

    pil_images = [ _TENSOR_TO_PIL(((image_tensors[i].cpu()+1)/2).clamp(0,1)) for i in range(image_tensors.size(0)) ]
    inputs = CLIP_PROCESSOR(images=pil_images, text=[prompt]*len(pil_images), return_tensors="pt", padding=True)
    inputs = {k:v.to(DEVICE) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = CLIP_MODEL(**inputs)
    img_emb, txt_emb = outputs.image_embeds, outputs.text_embeds
    img_norm = img_emb / (img_emb.norm(dim=-1, keepdim=True)+1e-12)
    txt_norm = txt_emb / (txt_emb.norm(dim=-1, keepdim=True)+1e-12)

    scores=[]
    for b in range(img_emb.size(0)):
        sdict={}
        if "cosine" in weights: sdict["cosine"]=1.0-float(torch.matmul(img_norm[b:b+1],txt_norm[b:b+1].t()).item())
        if "dot" in weights: sdict["dot"]=-float(torch.matmul(img_emb[b:b+1],txt_emb[b:b+1].t()).item())
        if "l2" in weights: sdict["l2"]=float(torch.norm(img_emb[b]-txt_emb[b],p=2).item())
        if "l1" in weights: sdict["l1"]=float(torch.norm(img_emb[b]-txt_emb[b],p=1).item())
        tw=sum(weights.values()) or 1.0
        combined=sum(sdict[k]*weights[k] for k in sdict)/tw
        scores.append(combined)
    return torch.tensor(scores, device=image_tensors.device)


# ====== heat_map ======
def heat_map(
    output: torch.Tensor,
    target: torch.Tensor,
    FE: torch.nn.Module,
    config,
    *,
    return_fid: bool = False,
    fid_metric: Optional[FrechetInceptionDistance] = None,
    normalize_feats: bool = True,
    gauss_sigma: Optional[float] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
    device = torch.device(getattr(config.model, "device", "cpu"))
    gauss_sigma = GAUSS_SIG_DEFAULT if gauss_sigma is None else float(gauss_sigma)
    kernel_size = _compute_kernel_size_for_sigma(gauss_sigma)

    output = output.to(device)
    target = target.to(device)

    # Pixel distances
    l1_d = torch.mean(torch.abs(output - target), dim=1, keepdim=True)
    l2_d = torch.mean((output - target) ** 2, dim=1, keepdim=True)

    # Feature distance
    f_d = feature_distance(output, target, FE, config, normalize=normalize_feats).to(device)

    # SSIM
    output_norm = (output + 1.0) / 2.0
    target_norm = (target + 1.0) / 2.0
    try:
        ssim_map = kornia_ssim(output_norm, target_norm, window_size=11)
        if ssim_map.dim() == 4:
            ssim_map_ch = ssim_map
        else:
            ssim_map_ch = ssim_map.unsqueeze(1)
        ssim_diff = 1.0 - ssim_map_ch.mean(dim=1, keepdim=True)
    except Exception:
        ssim_diff = l1_d.clone()

    # ===== 改良 FID =====
    fid_score = None
    fid_map = torch.zeros_like(l1_d, device=device)
    if return_fid:
        try:
            with torch.no_grad():
                # 使用 FE 提取特徵
                transform = _make_normalize_transform(device)
                f_out = FE(transform(output_norm))
                f_tgt = FE(transform(target_norm))

                if isinstance(f_out, list): f_out = f_out[-1]
                if isinstance(f_tgt, list): f_tgt = f_tgt[-1]

                # 攤平計算 mean / cov
                B, C, H, W = f_out.shape
                f_out = f_out.view(B, C, -1).mean(dim=-1)
                f_tgt = f_tgt.view(B, C, -1).mean(dim=-1)

                mu1, mu2 = f_out.mean(dim=0), f_tgt.mean(dim=0)
                sigma1, sigma2 = torch.cov(f_out.T), torch.cov(f_tgt.T)

                diff = mu1 - mu2
                diff_sq = diff.dot(diff)
                # trace(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
                covmean = torch.linalg.sqrtm((sigma1 @ sigma2).cpu().numpy())
                covmean = torch.tensor(covmean.real, device=device)
                fid_score = float((diff_sq + torch.trace(sigma1 + sigma2 - 2 * covmean)).item())
                fid_map = torch.full_like(l1_d, fill_value=fid_score, device=device)
        except Exception as e:
            logging.warning(f"[anomaly_utils] 改良版 FID 計算失敗: {e}")
            fid_score = 0.0
            fid_map = torch.zeros_like(l1_d, device=device)

    # ratio scaling
    def _safe_max(t: torch.Tensor) -> float:
        try: return float(torch.amax(t).detach().cpu().item())
        except Exception: return 0.0

    max_fd = _safe_max(f_d)
    if max_fd <= EPS:
        max_fd = max(_safe_max(l1_d), _safe_max(l2_d), _safe_max(ssim_diff), EPS)

    ratio_l1   = max_fd / (max(_safe_max(l1_d), EPS))
    ratio_l2   = max_fd / (max(_safe_max(l2_d), EPS))
    ratio_ssim = max_fd / (max(_safe_max(ssim_diff), EPS))
    ratio_fid  = max_fd / (max(fid_score, EPS)) if return_fid else 0.0

    v_l1 = float(getattr(config.model, "v", 1.0))
    anomaly_map = (
        W_FD * f_d +
        W_L1 * v_l1 * ratio_l1 * l1_d +
        W_L2 * ratio_l2 * l2_d +
        W_SSIM * ratio_ssim * ssim_diff +
        W_FID * ratio_fid * fid_map
    )

    # CLIPAD
    B = output.shape[0]
    if W_CLIPAD > 0 and _CLIP_ENABLED:
        clip_scores = _clip_batch_scores(output, prompt=CLIP_PROMPT, weights=CLIP_WEIGHTS)
        clip_scores_map = clip_scores.view(B,1,1,1)
        anomaly_map += W_CLIPAD * clip_scores_map

    anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), (gauss_sigma, gauss_sigma))
    anomaly_map = torch.sum(anomaly_map, dim=1, keepdim=True)

    if return_fid: return anomaly_map, fid_score
    return anomaly_map


# ====== feature_distance ======



# ====== patchify ======
def patchify(features: torch.Tensor, patchsize: int = 3, stride: int = 1) -> torch.Tensor:
    B, C, H, W = features.shape
    pad = (patchsize-1)//2
    uf = torch.nn.Unfold(kernel_size=patchsize, stride=stride, padding=pad)
    patches = uf(features)
    patches = patches.view(B, C, patchsize*patchsize, -1)
    pooled = patches.mean(dim=2)
    return pooled.view(B, C, H, W)
