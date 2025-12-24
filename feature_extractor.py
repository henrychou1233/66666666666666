# -*- coding: utf-8 -*-
"""
feature_extractor.py — Domain Adaptation (Cross-sample Version)
將 c、d 改為從其他 good 影像 sample 而來，實現跨樣本對齊 (cross-sample alignment)
"""

import os, math, logging, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from kornia.metrics import ssim as kornia_ssim
except Exception:
    kornia_ssim = None

from dataset import Dataset_maker
from unet import *
from resnet import *
from reconstruction import Reconstruction

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ======================================================
# ⚙️ Switches
# ======================================================
RESUME = False
USE_WARMUP = True
WARMUP_EPOCHS = 3
SAVE_EVERY_EPOCH = 1
USE_PENALTY = False
PENALTY_WEIGHT = 0.1

LOSS_WEIGHTS = {
    "l1": 0.1,
    "l2": 0.1,
    "cosine": 1.0,
    "kl": 0.1,
    "ssim": 0.1,
    "penalty": PENALTY_WEIGHT
}

# ======================================================
# 🔍 找最新 feat checkpoint
# ======================================================
def get_latest_checkpoint(path):
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.startswith("feat") and f != "feat_best"]
    if not files:
        return None
    latest = max(files, key=lambda x: int(''.join([c for c in x if c.isdigit()]) or 0))
    return os.path.join(path, latest)


# ======================================================
# 儲存 / 載入 checkpoint
# ======================================================
def save_checkpoint(path, feature_extractor, optimizer, scheduler, epoch, best_loss, loss_history):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "feature_extractor": feature_extractor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
        "loss_history": loss_history,
        "loss_weights": LOSS_WEIGHTS
    }
    torch.save(state, path)
    logging.info(f"[save] checkpoint saved at {path}")


def load_checkpoint(path, feature_extractor, optimizer, scheduler, device):
    global LOSS_WEIGHTS
    if not os.path.exists(path):
        logging.warning(f"⚠️ Checkpoint not found: {path}")
        return 0, float("inf"), []

    state = torch.load(path, map_location=device)
    fname = os.path.basename(path)

    if isinstance(state, dict) and "feature_extractor" in state:
        feature_extractor.load_state_dict(state["feature_extractor"], strict=False)
        if optimizer and "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and "scheduler" in state and state["scheduler"]:
            scheduler.load_state_dict(state["scheduler"])
        if "loss_weights" in state:
            LOSS_WEIGHTS.update(state["loss_weights"])
            logging.info(f"🔁 Restored LOSS_WEIGHTS: {LOSS_WEIGHTS}")
        start_epoch = state.get("epoch", 0) + 1
        best_loss = state.get("best_loss", float("inf"))
        loss_history = state.get("loss_history", [])
        logging.info(f"✅ Resume success from {path} (new-format checkpoint)")
    else:
        feature_extractor.load_state_dict(state, strict=False)
        digits = ''.join([c for c in fname if c.isdigit()])
        start_epoch = int(digits) if digits else 0
        best_loss, loss_history = float("inf"), []
        logging.info(f"⚙️ Loaded old-format checkpoint (state_dict only): {path}")

    return start_epoch, best_loss, loss_history


# ======================================================
# Optimizer
# ======================================================
def get_optimizer(model, lr=1e-4, wd=0.01, config=None):
    if config is not None:
        lr = float(getattr(config.model, "DA_lr", lr))
        wd = float(getattr(config.model, "DA_weight_decay", wd))
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": wd, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ])


# ======================================================
# Scheduler
# ======================================================
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-7, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            return [b * (e + 1) / max(1, self.warmup_epochs) for b in self.base_lrs]
        p = (e - self.warmup_epochs) / float(max(1, self.max_epochs - self.warmup_epochs))
        return [self.min_lr + (b - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))
                for b in self.base_lrs]


# ======================================================
# Loss functions
# ======================================================
def compute_losses(f1, f2, config):
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    B = f1.shape[0]
    f1f, f2f = f1.view(B, -1), f2.view(B, -1)
    l1 = (f1f - f2f).abs().mean()
    l2 = ((f1f - f2f) ** 2).mean()
    cos_loss = 1 - cos_sim(f1f, f2f).mean()

    try:
        log_q = F.log_softmax(f2f, dim=1)
        p = F.softmax(f1f, dim=1)
        kl = F.kl_div(log_q, p, reduction='batchmean')
    except:
        kl = torch.tensor(0.0, device=f1.device)

    ssim_loss = torch.tensor(0.0, device=f1.device)
    if kornia_ssim is not None:
        try:
            ssim_loss = 1.0 - kornia_ssim(f1, f2, window_size=3).mean()
        except:
            pass

    penalty = (f1f ** 2).mean() if USE_PENALTY else torch.tensor(0.0, device=f1.device)
    total = (
        LOSS_WEIGHTS["l1"] * l1 +
        LOSS_WEIGHTS["l2"] * l2 +
        LOSS_WEIGHTS["cosine"] * cos_loss +
        LOSS_WEIGHTS["kl"] * kl +
        LOSS_WEIGHTS["ssim"] * ssim_loss +
        LOSS_WEIGHTS["penalty"] * penalty
    )
    return total


def loss_function(a, b, c, d, config):
    feats_a = [a] if not isinstance(a, (list, tuple)) else a
    feats_b = [b] if not isinstance(b, (list, tuple)) else b
    feats_c = [c] if not isinstance(c, (list, tuple)) else c
    feats_d = [d] if not isinstance(d, (list, tuple)) else d
    n_layers = min(len(feats_a), len(feats_b), len(feats_c), len(feats_d))
    if n_layers == 0:
        return torch.tensor(0.0, device=config.model.device)
    λ = getattr(config.model, "DLlambda", 1.0)
    total = 0.0
    for i in range(n_layers):
        total += (
            compute_losses(feats_a[i], feats_b[i], config) +
            compute_losses(feats_b[i], feats_c[i], config) * λ +
            compute_losses(feats_a[i], feats_d[i], config) * λ
        )
    return torch.nan_to_num(total)


# ======================================================
# 建立 Feature Extractor
# ======================================================
def build_feature_extractor(config, use_latent=False):
    in_channels = 4 if use_latent else getattr(config.data, "input_channel", 3)
    backbone = getattr(config.model, "feature_extractor", "wide_resnet101_2")
    if backbone == "resnet50":
        model = resnet50(pretrained=True)
    else:
        model = wide_resnet101_2(pretrained=True)
    if model.conv1.in_channels != in_channels:
        model.conv1 = nn.Conv2d(
            in_channels, model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size, stride=model.conv1.stride,
            padding=model.conv1.padding, bias=model.conv1.bias is not None
        )
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    return model


# ======================================================
# Domain Adaptation (Cross-sample)
# ======================================================
def domain_adaptation(unet, config, fine_tune: bool):
    feature_extractor = build_feature_extractor(config)
    frozen_feature_extractor = build_feature_extractor(config)

    for p in feature_extractor.parameters():
        p.requires_grad = True
    feature_extractor.train()

    for p in frozen_feature_extractor.parameters():
        p.requires_grad = False
    frozen_feature_extractor.eval()

    feature_extractor = torch.nn.DataParallel(feature_extractor).to(config.model.device)
    frozen_feature_extractor = torch.nn.DataParallel(frozen_feature_extractor).to(config.model.device)

    train_dataset = Dataset_maker(root=config.data.data_dir, category=config.data.category, config=config, is_train=True)
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=max(1, int(getattr(config.data, "DA_batch_size", 8))),
        shuffle=True, num_workers=max(0, int(getattr(config.model, "num_workers", 4))),
        drop_last=True
    )

    checkpoint_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, config.data.category)
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = get_optimizer(feature_extractor, config=config)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, getattr(config.model, "DA_epochs", 10)) if USE_WARMUP else None
    reconstruction = Reconstruction(unet, config, feature_extractor=feature_extractor)
    DA_epochs = int(getattr(config.model, "DA_epochs", 10))

    start_epoch, best_loss, loss_history = (0, float("inf"), [])
    latest_ckpt = get_latest_checkpoint(checkpoint_dir)
    if RESUME and latest_ckpt:
        start_epoch, best_loss, loss_history = load_checkpoint(latest_ckpt, feature_extractor, optimizer, scheduler, config.model.device)

    plt.figure()
    for epoch in range(start_epoch, DA_epochs):
        total_loss = 0.0
        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"DA Epoch {epoch+1}/{DA_epochs}")

        for _, batch in pbar:
            imgs = batch[0].to(config.model.device)
            B = imgs.size(0)
            if B < 1:
                continue

            # 隨機打亂 index 取得不同樣本 (cross-sample pairing)
            rand_idx = torch.randperm(B)
            target = imgs
            other_imgs = imgs[rand_idx]

            # a = target, b = 重建(a)
            x0 = reconstruction(target, target, float(getattr(config.model, "w_DA", 1.0)))[-1]
            x0 = torch.nan_to_num(x0)

            # c = frozen_FE(重建的其他 good)
            x0_others = reconstruction(other_imgs, other_imgs, float(getattr(config.model, "w_DA", 1.0)))[-1]
            reconst_frozen_other = frozen_feature_extractor(x0_others)

            # d = frozen_FE(其他 good 原圖)
            frozen_target_other = frozen_feature_extractor(other_imgs)

            # feature extractions
            reconst_fe = feature_extractor(x0)
            target_fe = feature_extractor(target)

            # loss:  (a,b) + λ(b,c) + λ(a,d)
            loss = loss_function(reconst_fe, target_fe, reconst_frozen_other, frozen_target_other, config)

            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        avg_loss = total_loss / max(1, len(trainloader))
        logging.info(f"DA Epoch {epoch+1}/{DA_epochs} | Avg Loss {avg_loss:.6f}")
        if scheduler:
            scheduler.step()
        loss_history.append(avg_loss)

        if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
            save_checkpoint(os.path.join(checkpoint_dir, f"feat{epoch+1}"), feature_extractor, optimizer, scheduler, epoch, best_loss, loss_history)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(feature_extractor.state_dict(), os.path.join(checkpoint_dir, "feat_best"))
            logging.info(f"✅ New best loss: {best_loss:.6f} | feat_best saved")

        plt.clf()
        plt.title(f"DA Training Loss - {config.data.category}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.plot(range(1, len(loss_history)+1), loss_history, marker='o', color='b')
        plt.savefig(os.path.join(checkpoint_dir, f"{config.data.category}_da_loss.png"))

    plt.close()
    return feature_extractor
