# -*- coding: utf-8 -*-
# clipad2.py
# WinCLIP+ Segmentation 版：
# - Classification: Image AUROC (Top-k pooling, 保證分數越大越異常)
# - Segmentation: Pixel AUROC, PRO
# - Few-shot Normal Support 可選
# - 多尺度 sliding-window feature extraction
# - 含 tqdm 進度條
# - 輸出 CSV: class, image(relative path), score

import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import pandas as pd
import open_clip   # pip install open_clip_torch
from skimage import measure

# ================== 全域設定 ==================
MVTEC2_CLASSES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


BASE_DIR = Path("/mnt/c/Users/anywhere4090/Desktop/0902 finalcode/dataset/mvtec2")
OUT_BASE_DIR = Path("/mnt/c/Users/anywhere4090/Desktop/0902 finalcode")
RESULTS_CSV = OUT_BASE_DIR / "mvtec2_winclip_plus_seg_results.csv"
SCORES_CSV = OUT_BASE_DIR / "winclip_scores.csv"

IMGSIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Backbone 可選: "ViT-B-16" / "ViT-L-14"
BACKBONE = "ViT-L-14"
PRETRAINED = "laion400m_e32" if BACKBONE == "ViT-B-16" else "laion2b_s32b_b82k"

# Window/patch 設定
WINDOW_SIZES = [32, 48, 64]
STRIDE = 16

# Top-k pooling 設定
TOPK_RATIO = 0.05  # 取前 5%

# Few-shot Support
SUPPORT_MAX = 20  # 可調：取多少張 train/good 作 support

# ================== CPE Prompt ==================
STATE_WORDS_NORMAL = ["flawless", "intact", "perfect", "clean", "good"]
STATE_WORDS_ANOM = ["broken", "cracked", "damaged", "scratched", "defective", "faulty"]
TEMPLATES = [
    "a photo of a {}",
    "a cropped photo of a {}",
    "a photo of a {} for visual inspection"
]

def build_cpe_prompts(cls_name):
    normal_prompts = [t.format(w + " " + cls_name) for w in STATE_WORDS_NORMAL for t in TEMPLATES]
    anomaly_prompts = [t.format(w + " " + cls_name) for w in STATE_WORDS_ANOM for t in TEMPLATES]
    return normal_prompts, anomaly_prompts

# ================== 建立 Support Embeddings ==================
def build_support_embeddings(model, preprocess, cls, max_support=SUPPORT_MAX):
    support_dir = BASE_DIR / cls / "train" / "good"
    support_imgs = sorted(list(support_dir.glob("*.png")))[:max_support]
    support_embs = []
    for img_path in support_imgs:
        img = Image.open(img_path).convert("RGB").resize((IMGSIZE, IMGSIZE))
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        support_embs.append(emb)
    return torch.cat(support_embs, dim=0) if len(support_embs) > 0 else None

# ================== WinCLIP+ 特徵抽取 ==================
def extract_winclip_plus_features(model, preprocess, image, normal_prompts, anomaly_prompts,
                                  emb_norm_support=None, window_size=32, stride=16):
    W, H = image.size
    scores = np.zeros((H, W))
    counts = np.zeros((H, W))

    with torch.no_grad():
        txt_norm = model.encode_text(open_clip.tokenize(normal_prompts).to(DEVICE))
        txt_anom = model.encode_text(open_clip.tokenize(anomaly_prompts).to(DEVICE))
        txt_norm = txt_norm / txt_norm.norm(dim=-1, keepdim=True)
        txt_anom = txt_anom / txt_anom.norm(dim=-1, keepdim=True)

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            crop = image.crop((x, y, x + window_size, y + window_size))
            crop_tensor = preprocess(crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                img_emb = model.encode_image(crop_tensor)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            sim_norm_text = (img_emb @ txt_norm.T).max().item()
            sim_norm_support = (img_emb @ emb_norm_support.T).max().item() if emb_norm_support is not None else -1e9
            sim_norm = max(sim_norm_text, sim_norm_support)

            sim_anom = (img_emb @ txt_anom.T).max().item()

            # 🔹 保證「越像異常 → 分數越大」
            anomaly_score = sim_anom - sim_norm

            scores[y:y + window_size, x:x + window_size] += anomaly_score
            counts[y:y + window_size, x:x + window_size] += 1

    return scores / (counts + 1e-6)

# ================== 計算 PRO ==================
def compute_pro(masks, heatmaps, num_th=50):
    pros = []
    for th in np.linspace(0, 1, num_th):
        bin_preds = (heatmaps >= th).astype(np.uint8)
        for mask, pred in zip(masks, bin_preds):
            label_mask = measure.label(mask, connectivity=2)
            regions = np.unique(label_mask)[1:]
            for r in regions:
                region = (label_mask == r)
                inter = (pred * region).sum()
                union = region.sum()
                if union > 0:
                    pros.append(inter / union)
    return np.mean(pros) if len(pros) > 0 else 0.0

# ================== 推理 ==================
def run_inference_winclip_plus_seg(cls, max_support=SUPPORT_MAX):
    model, _, preprocess = open_clip.create_model_and_transforms(
        BACKBONE, pretrained=PRETRAINED
    )
    model = model.to(DEVICE).eval()

    emb_norm_support = build_support_embeddings(model, preprocess, cls, max_support=max_support)

    test_root = BASE_DIR / cls / "test"
    gt_root = BASE_DIR / cls / "ground_truth"

    y_true, y_score = [], []
    masks_all, maps_all = [], []
    normal_prompts, anomaly_prompts = build_cpe_prompts(cls)

    all_imgs = []
    for sub in ["good"] + [d.name for d in test_root.iterdir() if d.is_dir() and d.name != "good"]:
        for img_path in (test_root / sub).glob("*.png"):
            all_imgs.append((img_path, sub))

    scores_dict = []

    for img_path, sub in tqdm(all_imgs, desc=f"[{cls}] 推理中 ({len(all_imgs)} imgs)", unit="img"):
        label = 0 if sub == "good" else 1
        img = Image.open(img_path).convert("RGB").resize((IMGSIZE, IMGSIZE))

        score_maps = []
        for ws in WINDOW_SIZES:
            score_map = extract_winclip_plus_features(
                model, preprocess, img, normal_prompts, anomaly_prompts,
                emb_norm_support=emb_norm_support,
                window_size=ws, stride=STRIDE
            )
            score_maps.append(score_map)

        final_map = np.mean(score_maps, axis=0)
        flat = final_map.flatten()
        k = max(1, int(len(flat) * TOPK_RATIO))
        topk = np.partition(flat, -k)[-k:]
        score = topk.mean()  # 🔹 保持「異常分數越大」

        y_true.append(label)
        y_score.append(score)

        # 存相對路徑
        rel_path = str(img_path.relative_to(BASE_DIR)).replace("\\", "/")
        scores_dict.append({"class": cls, "image": rel_path, "score": score})

        if label == 1:
            mask_path = gt_root / sub / img_path.name
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L").resize(final_map.shape[::-1])
                mask = (np.array(mask) > 127).astype(np.uint8)
                masks_all.append(mask)
                maps_all.append((final_map - final_map.min()) / (final_map.max() - final_map.min() + 1e-6))

    img_auroc = roc_auc_score(y_true, y_score)
    if len(masks_all) > 0:
        masks_all = np.array(masks_all)
        maps_all = np.array(maps_all)
        px_auroc = roc_auc_score(masks_all.flatten(), maps_all.flatten())
        pro = compute_pro(masks_all, maps_all)
    else:
        px_auroc, pro = np.nan, np.nan

    return img_auroc, px_auroc, pro, scores_dict

# ================== Pipeline 主程式 ==================
results, all_scores = [], []
for cls in MVTEC2_CLASSES:
    print(f"\n>>> Class: {cls} (Backbone={BACKBONE})")
    img_auroc, px_auroc, pro, scores_dict = run_inference_winclip_plus_seg(cls, max_support=SUPPORT_MAX)
    print(f"[RESULT] {cls} Image-AUROC={img_auroc:.4f}, Pixel-AUROC={px_auroc:.4f}, PRO={pro:.4f}")
    results.append({
        "class": cls,
        "backbone": BACKBONE,
        "support": SUPPORT_MAX,
        "image_auroc": img_auroc,
        "pixel_auroc": px_auroc,
        "pro": pro
    })
    all_scores.extend(scores_dict)

df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)
print(f"Results saved to {RESULTS_CSV}")

df_scores = pd.DataFrame(all_scores)
df_scores.to_csv(SCORES_CSV, index=False)
print(f"WinCLIP scores saved to {SCORES_CSV}")
