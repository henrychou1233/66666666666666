# -*- coding: utf-8 -*-
"""
app.py — Optimized Gradio Inference (No Domain Adaptation Re-training)
(只初始化一次模型，不重新訓練 DA，忽略 CLIP 錯誤)
"""

import os
import io
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from omegaconf import OmegaConf
import gradio as gr
import cv2
import matplotlib.pyplot as plt
import time
import logging

# =====================================================
# 基礎設定
# =====================================================
logging.basicConfig(level=logging.INFO, format="%(message)s")

from main import build_model
from reconstruction import Reconstruction
from anomaly_map import heat_map
from feature_extractor import build_feature_extractor

# =====================================================
# 🔧 模型初始化（只執行一次）
# =====================================================
cfg = OmegaConf.load('config.yaml')
cfg.data.category = 'one'  # 修改為你的類別
cfg.model.load_chp = 300
cfg.model.DA_chp = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===== 建立 Unet 模型 =====
unet = build_model(cfg)
unet = torch.nn.DataParallel(unet)

ckpt_folder = os.path.join(cfg.model.checkpoint_dir, cfg.data.category, str(cfg.model.load_chp))
if os.path.isfile(ckpt_folder):
    ckpt_path = ckpt_folder
else:
    ckpt_path = None
    for fn in os.listdir(ckpt_folder):
        if fn.endswith(('.pth', '.pt')):
            ckpt_path = os.path.join(ckpt_folder, fn)
            break
assert ckpt_path, f"❌ 未找到 checkpoint 檔案: {ckpt_folder}"

unet.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
unet.to(device).eval()

recon_m = Reconstruction(unet, cfg)

# ===== Domain Adaptation (只載入，不重建) =====
feat_folder = os.path.join(cfg.model.checkpoint_dir, cfg.data.category, f"feat{cfg.model.DA_chp}")
if os.path.isfile(feat_folder):
    feat_path = feat_folder
else:
    feat_path = None
    for fn in os.listdir(feat_folder):
        if fn.endswith(('.pth', '.pt')):
            feat_path = os.path.join(feat_folder, fn)
            break
assert feat_path, f"❌ 未找到 Domain Adaptation checkpoint: {feat_folder}"

# ✅ 只載入 DA 結構與權重，不進行訓練
fe = build_feature_extractor(cfg)
fe = torch.nn.DataParallel(fe).to(device)
fe.load_state_dict(torch.load(feat_path, map_location=device), strict=False)
fe.eval()

print(f"[INFO] ✅ 模型初始化完成 (U-Net + DomainAdaptation) on {device}")

# =====================================================
# 🧩 前處理函數
# =====================================================
def preprocess(image, img_size, device):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    img = image.convert('RGB')
    return tf(img).unsqueeze(0).to(device)

# =====================================================
# 📊 熱圖分析
# =====================================================
def analyze_heatmap(heatmap_img_pil, orig_img_pil, recon_img_pil, min_area=50.0):
    heatmap_color = np.array(heatmap_img_pil.convert('RGB'))[..., ::-1]
    orig = np.array(orig_img_pil.convert('RGB'))[..., ::-1]
    recon = np.array(recon_img_pil.convert('RGB'))[..., ::-1]
    heatmap_vis = heatmap_color.copy()

    heatmap_gray = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(heatmap_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    report = []
    for idx, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        report.append((idx, x, y, w, h, area))
        cv2.rectangle(heatmap_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(heatmap_vis, f"({x},{y})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if not report:
        analysis_txt = "沒有偵測到超過最小面積門檻的異常區域。"
    else:
        analysis_txt = "本次檢測共發現以下異常區域：\n"
        for idx, x, y, w, h, area in report:
            analysis_txt += f"・第{idx+1}個異常區域，位置({x},{y})，大小{w}×{h}，面積{area:.1f}\n"
        analysis_txt += f"\n總計偵測到 {len(report)} 個異常區域。"

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    images = [orig, recon, heatmap_color, heatmap_vis]
    titles = ["Original", "Reconstruction", "Raw Heatmap", "Annotated Heatmap"]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    summary_img = Image.open(buf)
    return summary_img, analysis_txt

# =====================================================
# 🚀 推論函數（不重建模型）
# =====================================================
def gradio_infer(img_pil):
    time_start = time.time()
    x = preprocess(img_pil, cfg.data.image_size, device)

    with torch.no_grad():
        x0_hat = recon_m(x, x, cfg.model.w)[-1]
        amap = heat_map(x0_hat, x, fe, cfg)

    # 正規化 anomaly map
    amap_min = amap.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    amap_max = amap.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    amap_norm = (amap - amap_min) / (amap_max - amap_min + 1e-8)
    amap_norm = amap_norm.clamp(0, 1)

    # 取得 anomaly score
    score = float(amap_norm.view(amap_norm.size(0), -1).mean(dim=1).item())
    threshold = 0.3
    pred_label = 'Anomalous' if score > threshold else 'Good'

    # 生成熱圖與重建圖
    amap_img = (amap_norm[0, 0].cpu().numpy() * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(amap_img)
    recon_np = ((x0_hat.squeeze(0).cpu().clamp(-1, 1) + 1) / 2 * 255).permute(1, 2, 0).byte().numpy()
    recon_img = Image.fromarray(recon_np)

    # 分析熱圖
    summary_img, analysis_txt = analyze_heatmap(heatmap_img, img_pil, recon_img)
    score_str = f"{score:.6f}"
    elapsed = time.time() - time_start

    info = (
        f"Anomaly score: {score_str} | Threshold: {threshold} | Prediction: {pred_label}\n"
        f"推論花費時間：{elapsed:.3f} 秒"
    )
    return pred_label, score_str, heatmap_img, recon_img, summary_img, analysis_txt, info

# =====================================================
# 🖼️ Gradio 介面
# =====================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Real-Time Diffusion Anomaly Inspector\n請上傳圖片，系統將進行異常檢測並輸出重建與熱圖分析。")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="上傳圖片", type="pil")
            btn = gr.Button("開始偵測")
        with gr.Column():
            pred = gr.Textbox(label="分類結果")
            score = gr.Textbox(label="異常分數")
            heatmap = gr.Image(label="Anomaly Map")
            recon = gr.Image(label="Reconstruction")

    gr.Markdown("## 🔍 異常區域分析（含輪廓、座標與面積報告）")
    with gr.Row():
        summary_img = gr.Image(label="異常總覽")
        analysis_txt = gr.Textbox(label="異常區域報告", lines=6)
    log = gr.Textbox(label="推論資訊", interactive=False)

    btn.click(fn=gradio_infer,
              inputs=[image_input],
              outputs=[pred, score, heatmap, recon, summary_img, analysis_txt, log],
              api_name="predict")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
