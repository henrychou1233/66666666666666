# ======================================================
# ddad.py — Full Metrics + Dynamic w + Config Dump (2025-10-24)
# ======================================================

import os
import time
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import datetime
import yaml
from typing import Any
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
from diffusers import AutoencoderKL
from sklearn.metrics import (
    confusion_matrix, f1_score, average_precision_score,
    precision_score, recall_score, roc_auc_score
)
import pandas as pd

# ======================================================
# 🧩 全域超參數設定
# ======================================================
# ======================================================
# 🧩 全域超參數設定 (Global Hyperparameters)
# ======================================================
USE_DYNAMIC_W: bool = True # 是否啟用 Dynamic w 機制 (決定是否根據 LOF/KNN 等分數動態調整重建引導權重 w)
ALGO_TYPE: str = 'lof' # 使用的異常檢測算法 (用於計算 Dynamic w 的原始分數 raw_scores)
                                    # 可選：'lof' (局部離群因子) / 'knn' (K近鄰距離) / 'iforest' (孤立森林) / 'ocsvm' (單類 SVM)

NEIGHBORS: int = 10                 # K近鄰算法 (如 LOF, KNN) 中，鄰居的數量 K
CONTAMINATION: float = 0         # 訓練集中預期的異常樣本比例 (用於 Isolation Forest, One-Class SVM 等)
RANDOM_STATE: int = 42              # 隨機種子 (用於確保結果可重現性)

DYNAMIC_W_ALPHA: float = 1.2          # 動態權重調整的放大係數 (決定 raw_scores 差異對 w 調整量的影響程度)
DYNAMIC_W_CLIP_RATIO: float = 1   # Dynamic w 調整量 (Delta) 的裁剪比例 (相對於 BASE_W 的最大調整範圍)
                                    # Delta 最大/最小值為 +/- (BASE_W * DYNAMIC_W_CLIP_RATIO)
DYNAMIC_W_MIN: float = 0.0          # 調整後的 w 最小值 (確保 w 不會過低)
DYNAMIC_W_MAX: float = 30          # 調整後的 w 最大值 (確保 w 不會過高)
BASE_W: float = 1.0                 # 引導權重 w 的基準值 (當 Dynamic w 調整量為零時所採用的值)
TRAIN_FEATURE_BATCH: int = 20       # 提取訓練集特徵時所使用的批次大小

# ======================================================
# 匯入異常檢測模型
# ======================================================
try:
    from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
except Exception:
    LocalOutlierFactor = NearestNeighbors = IsolationForest = OneClassSVM = None

# ======================================================
# 🧩 遞迴展開 Config 函式
# ======================================================
def flatten_config(cfg, prefix=""):
    items = {}
    for key in dir(cfg):
        if key.startswith("_"):
            continue
        try:
            val = getattr(cfg, key)
        except Exception:
            continue
        if hasattr(val, "__dict__"):
            try:
                items.update(flatten_config(val, f"{prefix}{key}."))
            except Exception:
                items[f"{prefix}{key}"] = str(val)
        else:
            items[f"{prefix}{key}"] = val
    return items

# ======================================================
# 🧠 DDAD 主類別
# ======================================================
class DDAD:
    def __init__(self, unet, config) -> None:
        self.config = config
        self.unet = unet
        self.device = getattr(config.model, "device", "cuda")

        # 測試資料載入
        self.test_dataset = Dataset_maker(
            root=getattr(config.data, "data_dir", ""),
            category=getattr(config.data, "category", ""),
            config=config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=getattr(config.data, "test_batch_size", 4),
            shuffle=False,
            num_workers=getattr(config.model, "num_workers", 0),
            drop_last=False,
        )

        # 保留原 transform
        self.transform = transforms.Compose([transforms.CenterCrop((224))])
        self.feature_extractor = None
        self.use_latent = getattr(config.model, "latent", False)
        self.vae = None

        self._build_feature_extractor_and_train_bank()
        self.reconstruction = Reconstruction(
            self.unet, self.config, feature_extractor=self.feature_extractor
        )

    # ======================================================
    # 建立特徵抽取器 + 訓練特徵庫
    # ======================================================
    def _build_feature_extractor_and_train_bank(self):
        global USE_DYNAMIC_W
        device = getattr(self.config.model, "device", "cuda")

        try:
            fe = build_feature_extractor(self.config, use_latent=self.use_latent)
        except Exception:
            fe = None
        self.feature_extractor = fe.to(device).eval() if fe else None

        # 若使用 latent，建立 VAE
        if self.use_latent:
            try:
                self.vae = AutoencoderKL.from_pretrained(
                    "CompVis/stable-diffusion-v1-4", subfolder="vae"
                ).to(device).eval()
            except Exception:
                self.vae = None

        # 準備訓練資料特徵庫
        train_dataset = Dataset_maker(
            root=getattr(self.config.data, "data_dir", ""),
            category=getattr(self.config.data, "category", ""),
            config=self.config,
            is_train=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=TRAIN_FEATURE_BATCH,
            shuffle=False,
            num_workers=max(0, getattr(self.config.model, "num_workers", 0)),
            drop_last=False,
        )

        train_feats = []
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Extract train features for dynamic w", unit="batch"):
                imgs = batch[0].to(device)
                model_input = imgs
                if self.use_latent and self.vae is not None:
                    try:
                        model_input = self.vae.encode(imgs).latent_dist.sample() * 0.18215
                    except Exception:
                        model_input = imgs
                feats = self.feature_extractor(model_input)
                if isinstance(feats, (list, tuple)):
                    feats = feats[0]
                if feats.dim() == 4:
                    feats = feats.view(feats.size(0), feats.size(1), -1).mean(dim=2)
                train_feats.append(feats.detach().cpu().numpy())

        self.train_features = np.concatenate(train_feats, axis=0) if train_feats else np.zeros((1, 1))
        self.model_dn, self.dn_baseline = None, 0.0

        # 初始化 dynamic w 模型
        if USE_DYNAMIC_W:
            try:
                if ALGO_TYPE == 'lof' and LocalOutlierFactor is not None:
                    lof = LocalOutlierFactor(n_neighbors=NEIGHBORS, novelty=True)
                    lof.fit(self.train_features)
                    train_scores = -lof.decision_function(self.train_features)
                    self.model_dn = lof

                elif ALGO_TYPE == 'knn' and NearestNeighbors is not None:
                    knn = NearestNeighbors(n_neighbors=NEIGHBORS)
                    knn.fit(self.train_features)
                    dists, _ = knn.kneighbors(self.train_features)
                    train_scores = np.mean(dists, axis=1)
                    self.model_dn = knn

                elif ALGO_TYPE == 'iforest' and IsolationForest is not None:
                    iforest = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
                    iforest.fit(self.train_features)
                    train_scores = -iforest.decision_function(self.train_features)
                    self.model_dn = iforest

                elif ALGO_TYPE == 'ocsvm' and OneClassSVM is not None:
                    ocsvm = OneClassSVM(nu=CONTAMINATION)
                    ocsvm.fit(self.train_features)
                    train_scores = -ocsvm.decision_function(self.train_features)
                    self.model_dn = ocsvm

                else:
                    USE_DYNAMIC_W = False
                    train_scores = np.zeros_like(self.train_features[:, 0])

                self.dn_baseline = float(np.mean(train_scores))
            except Exception as e:
                print(f"[WARN] Dynamic-w init failed: {e}")
                self.model_dn, self.dn_baseline = None, 0.0

    # ======================================================
    # 主流程：Inference + Metrics + Summary
    # ======================================================
    def __call__(self) -> Any:
        device = self.device
        feature_extractor = self.feature_extractor
        if feature_extractor:
            feature_extractor.eval()

        labels_list, predictions = [], []
        anomaly_map_list, gt_list = [], []
        reconstructed_list, forward_list = [], []

        use_dynamic = USE_DYNAMIC_W and (self.model_dn is not None)
        start_time, num_samples = time.time(), 0

        # === 推論階段 ===
        with torch.no_grad():
            for input, gt, labels in tqdm(self.testloader, desc="Inference", unit="batch"):
                input, gt = input.to(device), gt.to(device)
                num_samples += input.size(0)

                # --- 動態 w 計算 ---
                if use_dynamic and feature_extractor:
                    try:
                        feats = feature_extractor(input)
                        if isinstance(feats, (list, tuple)):
                            feats = feats[0]
                        if feats.dim() == 4:
                            feats = feats.view(feats.size(0), feats.size(1), -1).mean(dim=2)
                        feats_np = feats.cpu().numpy()

                        if ALGO_TYPE == 'lof':
                            raw_scores = -self.model_dn.decision_function(feats_np)
                        elif ALGO_TYPE == 'knn':
                            dists, _ = self.model_dn.kneighbors(feats_np)
                            raw_scores = np.mean(dists, axis=1)
                        elif ALGO_TYPE in ('iforest', 'ocsvm'):
                            raw_scores = -self.model_dn.decision_function(feats_np)
                        else:
                            raw_scores = np.zeros_like(feats_np[:, 0])

                        deltas = DYNAMIC_W_ALPHA * (raw_scores - self.dn_baseline)
                        deltas = np.clip(deltas, -BASE_W * DYNAMIC_W_CLIP_RATIO, BASE_W * DYNAMIC_W_CLIP_RATIO)

                        # 🔹 修改開始: 當 raw_scores > baseline (更異常) 時，減少權重
                        w_values = np.clip(BASE_W - deltas, DYNAMIC_W_MIN, DYNAMIC_W_MAX)
                        # 🔹 修改結束

                        w_dn = torch.tensor(w_values, device=device, dtype=torch.float32).view(-1, 1, 1, 1)
                    except Exception:
                        w_dn = torch.tensor(BASE_W, device=device).view(1, 1, 1, 1)
                else:
                    w_dn = torch.tensor(BASE_W, device=device).view(1, 1, 1, 1)

                # --- Reconstruction ---
                x0 = self.reconstruction(input, input, w_dn, is_dynamic=True)[-1]
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)
                if isinstance(anomaly_map, (tuple, list)):
                    anomaly_map = anomaly_map[0]

                anomaly_map, gt = self.transform(anomaly_map), self.transform(gt)
                forward_list.append(input.cpu())
                anomaly_map_list.append(anomaly_map.cpu())
                gt_list.append(gt.cpu())
                reconstructed_list.append(x0.cpu())

                for pred_map, label in zip(anomaly_map.detach().cpu(), labels):
                    labels_list.append(0 if label == 'good' else 1)
                    predictions.append(float(pred_map.max().item()))

        # ======================================================
        # ⏱️ 評估與輸出
        # ======================================================
        total_time = time.time() - start_time
        fps = num_samples / total_time if total_time > 0 else float('inf')
        latency_ms = (total_time / num_samples) * 1000.0 if num_samples > 0 else float('nan')

        # ------------------ LOG & CSV DUMP ------------------
        try:
            os.makedirs("logs", exist_ok=True)
            cat = getattr(self.config.data, "category", "unknown")
            csv_path = f"logs/anomaly_scores_{cat}.csv"

            amaps = torch.cat(anomaly_map_list).cpu().numpy()
            N = amaps.shape[0]
            maps_flat = amaps.reshape(N, -1)

            score_max = maps_flat.max(axis=1)
            score_mean = maps_flat.mean(axis=1)
            score_median = np.median(maps_flat, axis=1)
            score_p95 = np.percentile(maps_flat, 95, axis=1)
            k = min(10, maps_flat.shape[1])
            score_topk = np.sort(maps_flat, axis=1)[:, -k:].mean(axis=1)

            per_min = maps_flat.min(axis=1, keepdims=True)
            per_ptp = maps_flat.ptp(axis=1, keepdims=True) + 1e-12
            per_norm = (maps_flat - per_min) / per_ptp
            score_p95_per = np.percentile(per_norm, 95, axis=1)
            area_frac = (per_norm > 0.5).sum(axis=1) / per_norm.shape[1]

            labels_np = np.array(labels_list).astype(int)

            df = pd.DataFrame({
                "idx": np.arange(N),
                "label": labels_np,
                "score_max": score_max,
                "score_mean": score_mean,
                "score_median": score_median,
                "score_p95_raw": score_p95,
                "score_topk_mean": score_topk,
                "score_p95_per_image": score_p95_per,
                "area_frac_per_image": area_frac
            })

            df.to_csv(csv_path, index=False)
            print(f"[LOG] Saved per-image anomaly scores to: {csv_path}")

            def safe_auc(y, s):
                if len(np.unique(y)) < 2:
                    return float('nan')
                try:
                    return roc_auc_score(y, s)
                except Exception:
                    return float('nan')

            aucs = {
                "max_raw": safe_auc(labels_np, df["score_max"].values),
                "mean_raw": safe_auc(labels_np, df["score_mean"].values),
                "p95_raw": safe_auc(labels_np, df["score_p95_raw"].values),
                "topk_mean": safe_auc(labels_np, df["score_topk_mean"].values),
                "p95_per_image": safe_auc(labels_np, df["score_p95_per_image"].values),
            }

            print("\n[DIAGNOSTIC] Image-level AUCs (various aggregations):")
            for k, v in aucs.items():
                print(f"  {k:15s} : {v:.4f}")

            print("\n[DIAGNOSTIC] Sample rows (first 12):")
            print(df.head(12).to_string(index=False))

            print("\n[DIAGNOSTIC] Per-sample (idx, label, score_max, score_p95_per_image, score_topk_mean):")
            for _, row in df.iterrows():
                print(
                    f"  idx={int(row['idx']):04d} | label={int(row['label'])} | "
                    f"max={row['score_max']:.6f} | p95_per={row['score_p95_per_image']:.6f} | "
                    f"topk_mean={row['score_topk_mean']:.6f}"
                )

            th = 0.5
            pred_labels = (df["score_p95_per_image"].values >= float(th)).astype(int)
            fn_idx = df[(df["label"] == 1) & (pred_labels == 0)]["idx"].values
            fp_idx = df[(df["label"] == 0) & (pred_labels == 1)]["idx"].values
            print(f"\n[DIAGNOSTIC] Using threshold={th:.4f} -> FN count={len(fn_idx)}, FP count={len(fp_idx)}")
            print("  Example FN indices:", fn_idx[:10])
            print("  Example FP indices:", fp_idx[:10])

            debug_dir = os.path.join("logs", f"debug_{cat}")
            os.makedirs(debug_dir, exist_ok=True)
            import torchvision.utils as vutils

            def save_img_tensor(tensor, path):
                t = tensor.clone()
                if t.min() < -0.5:
                    t = (t + 1) / 2
                t = t.clamp(0, 1)
                vutils.save_image(t, path)

            maxsave = 20
            for i in fn_idx[:maxsave]:
                try:
                    inp = forward_list[int(i)]
                    amap = anomaly_map_list[int(i)]
                    gt = gt_list[int(i)]
                    save_img_tensor(inp, os.path.join(debug_dir, f"fn_{i:04d}_input.png"))
                    save_img_tensor(amap.squeeze(0), os.path.join(debug_dir, f"fn_{i:04d}_amap.png"))
                    save_img_tensor(gt.squeeze(0), os.path.join(debug_dir, f"fn_{i:04d}_gt.png"))
                except Exception:
                    pass

            for i in fp_idx[:maxsave]:
                try:
                    inp = forward_list[int(i)]
                    amap = anomaly_map_list[int(i)]
                    gt = gt_list[int(i)]
                    save_img_tensor(inp, os.path.join(debug_dir, f"fp_{i:04d}_input.png"))
                    save_img_tensor(amap.squeeze(0), os.path.join(debug_dir, f"fp_{i:04d}_amap.png"))
                    save_img_tensor(gt.squeeze(0), os.path.join(debug_dir, f"fp_{i:04d}_gt.png"))
                except Exception:
                    pass

            print(f"[LOG] Saved FN/FP debug images to: {debug_dir}")

        except Exception as e_log:
            print(f"[WARN] Diagnostic logging failed: {e_log}")

        # ======================================================
        # 📊 Metrics Summary
        # ======================================================
        metric = Metric(labels_list, predictions, anomaly_map_list, gt_list, self.config)
        metric.optimal_threshold()
        thresh = getattr(metric, "threshold", 0.5)

        try:
            img_auroc = metric.image_auroc()
        except:
            img_auroc = float('nan')
        try:
            pro_score = metric.pixel_pro()
        except:
            pro_score = float('nan')
        ap_image = average_precision_score(labels_list, predictions) if len(set(labels_list)) > 1 else float('nan')

        try:
            all_gt = torch.cat(gt_list).flatten().cpu().numpy()
            all_pred = torch.cat(anomaly_map_list).flatten().cpu().numpy()
            px_auroc = metric.pixel_auroc()
            ap_pixel = average_precision_score(all_gt > 0.5, all_pred) if len(np.unique(all_gt)) > 1 else float('nan')
        except Exception:
            px_auroc, ap_pixel = float('nan'), float('nan')

        pred_labels = (np.array(predictions) >= float(thresh)).astype(int)
        try:
            f1_img = f1_score(labels_list, pred_labels)
            precision_img = precision_score(labels_list, pred_labels)
            recall_img = recall_score(labels_list, pred_labels)
        except:
            f1_img = precision_img = recall_img = float('nan')

        dice = (2 * precision_img * recall_img) / (precision_img + recall_img + 1e-6)
        iou = dice / (2 - dice)

        try:
            cm = confusion_matrix(labels_list, pred_labels)
        except:
            cm = np.array([[]])

        single_image_time = total_time / num_samples if num_samples > 0 else float('nan')

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epochs = getattr(self.config.model, "epochs", "na")
        da_epochs = getattr(self.config.model, "da_epochs", "na")

        # === 防呆參數導入 ===
        try:
            from anomaly_map import W_FD, W_L1, W_L2, W_SSIM, W_FID, W_CLIPAD, GAUSS_SIG_DEFAULT, EPS
            from reconstruction import GUIDANCE_W, RAW_W, FEAT_W, CLIP_W
        except Exception:
            W_FD = W_L1 = W_L2 = W_SSIM = W_FID = W_CLIPAD = GAUSS_SIG_DEFAULT = EPS = "N/A"
            GUIDANCE_W = RAW_W = FEAT_W = CLIP_W = "N/A"

        load_chp = getattr(self.config.model, "load_chp", "N/A")
        da_chp = getattr(self.config.model, "DA_chp", "N/A")

        # ======================================================
        # 📝 Summary
        # ======================================================
        summary = []
        cfg_path = "C:/Users/anywhere4090/Desktop/0902 finalcode/config.yaml"
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()
            summary.append("\n📘 Full Config YAML Dump")
            summary.append("──────────────────────────────────────────────")
            summary.append(yaml_content)
            summary.append("──────────────────────────────────────────────\n")
        else:
            summary.append("\n📘 Full Config Dump (Flattened)")
            summary.append("──────────────────────────────────────────────")
            all_cfg = flatten_config(self.config)
            for k, v in all_cfg.items():
                summary.append(f"{k:<40} : {v}")
            summary.append("──────────────────────────────────────────────\n")

        summary.append(f"\n\n╔════════════════════════════════════════════════════════╗")
        summary.append(f"║              🧪 DDAD Experiment Summary                ║")
        summary.append(f"║                [{timestamp}]                           ║")
        summary.append(f"╚════════════════════════════════════════════════════════╝\n")

        summary.append("📊 Evaluation Metrics Summary")
        summary.append("──────────────────────────────────────────────")
        summary.append(f"Category           : {getattr(self.config.data, 'category', 'N/A')}")
        summary.append(f"Image AUROC        : {img_auroc:.4f}")
        summary.append(f"Pixel AUROC        : {px_auroc:.4f}")
        summary.append(f"PRO (Region Overlap): {pro_score}")
        summary.append(f"Avg Precision (Img): {ap_image:.4f}")
        summary.append(f"Avg Precision (Px) : {ap_pixel:.4f}")
        summary.append(f"F1 Score (Img)     : {f1_img:.4f}")
        summary.append(f"Precision (Img)    : {precision_img:.4f}")
        summary.append(f"Recall (Img)       : {recall_img:.4f}")
        summary.append(f"Dice Coefficient   : {dice:.4f}")
        summary.append(f"IoU                : {iou:.4f}")
        summary.append(f"FPS                : {fps:.2f}")
        summary.append(f"Latency (ms/img)   : {latency_ms:.2f}")
        summary.append(f"Single Image Time  : {single_image_time:.4f}s")
        summary.append("──────────────────────────────────────────────")
        summary.append(f"Confusion Matrix:\n{cm}")
        summary.append("════════════════════════════════════════════════════════")

        # === 寫入 log ===
        with open("mtkaide.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(summary))
        print("\n".join(summary))

        if getattr(self.config.metrics, "visualisation", False):
            os.makedirs('results', exist_ok=True)
            visualize(
                torch.cat(forward_list),
                torch.cat(reconstructed_list),
                torch.cat(gt_list),
                (torch.cat(anomaly_map_list) > thresh).float(),
                torch.cat(anomaly_map_list),
                getattr(self.config.data, "category", "unknown")
            )
