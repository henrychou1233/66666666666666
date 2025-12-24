# -*- coding: utf-8 -*-
"""
train.py — Stable Adaptive Optimizer + WarmupHoldCosine v4
(改良動態WD，移除動態LR微調 + 梯度防爆 + WarmupHoldCosine)
"""

import os
import math
import logging
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import Dataset_maker
from loss import get_loss, reset_adaptive_state, print_adaptive_weights

try:
    from loss import LOSS_WEIGHTS
except ImportError:
    LOSS_WEIGHTS = {"raw": 1.0, "feat": 1, "penalty": 0}

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ======================================================
# CONFIG SWITCHES
# ======================================================
RESUME = False
SAVE_EVERY_EPOCH = 10
OPTIMIZER_TYPE = 'adamw_adaptive_dynamicwd' # ***保留原始名稱***
GRAD_CLIP_NORM = 3
USE_WARMUP = True
WARMUP_EPOCHS = 8
USE_REDUCEONPLATEAU = True
REDUCE_FACTOR = 0.6
REDUCE_PATIENCE = 3
MIN_LR = 5e-6
SAVE_BEST_ONLY = False


# ======================================================
# 🔍 找最新數字 checkpoint
# ======================================================
def get_latest_checkpoint(path):
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.isdigit()]
    if not files:
        return None
    latest = max(files, key=lambda x: int(x))
    return os.path.join(path, latest)


# ======================================================
# 🔹 AdamW_Adaptive_DynamicWD 改良版 v2 (移除 LR 調整)
# ======================================================
class AdamW_Adaptive_DynamicWD(torch.optim.AdamW): # ***保留原始名稱***
    def __init__(self, params, lr=1e-4, weight_decay=0.01,
                 min_wd=1e-7, max_wd=0.25,
                 loss_target=0.10, loss_sensitivity=0.25,
                 adapt_by_grad=True, ema_decay=0.9,
                 grad_norm_clip=1e6, wd_change_limit=0.3):
        super().__init__(params, lr=lr, weight_decay=weight_decay)
        self.min_wd = min_wd
        self.max_wd = max_wd
        self.loss_target = loss_target
        self.loss_sensitivity = loss_sensitivity
        self.adapt_by_grad = adapt_by_grad
        self.ema_decay = ema_decay
        self.grad_norm_clip = grad_norm_clip
        self.wd_change_limit = wd_change_limit
        self.loss_ema = None
        self.grad_ratio_smooth = 1.0
        for g in self.param_groups:
            g.setdefault("wd_base", float(g.get("weight_decay", 0.0)))
            g.setdefault("wd_ema", float(g.get("weight_decay", 0.0)))

    def step_dynamic_wd(self, epoch, loss_val):
        """依據平滑後的 loss 變化自動調整 weight decay (已移除 LR 調整)"""
        if isinstance(loss_val, torch.Tensor):
            loss_val = float(loss_val.detach().cpu().item())

        # === 平滑 loss ===
        if self.loss_ema is None:
            self.loss_ema = loss_val
        self.loss_ema = self.ema_decay * self.loss_ema + (1 - self.ema_decay) * loss_val

        loss_diff = self.loss_ema - self.loss_target
        loss_factor = 1.0 + self.loss_sensitivity * max(-2.0, min(4.0, loss_diff))
        loss_factor = max(0.5, min(2.5, loss_factor))

        for group in self.param_groups:
            wd_base = float(group.get("wd_base", 0.0))
            old_wd_ema = float(group.get("wd_ema", wd_base))
            if wd_base <= 0.0:
                continue

            # === 計算平滑 grad/param 比例 (r_grad) ===
            param_norm, grad_norm = 0.0, 0.0
            for p in group["params"]:
                if p.grad is not None:
                    grad_norm += float(torch.sum((p.grad.detach()) ** 2).item())
                param_norm += float(torch.sum((p.detach()) ** 2).item())
            param_norm = math.sqrt(max(param_norm, 1e-8))
            grad_norm = math.sqrt(max(grad_norm, 1e-8))
            ratio = grad_norm / param_norm if self.adapt_by_grad else 1.0

            # === 梯度爆炸防禦（自動回退）===
            # 此處保留原始代碼，用於在優化器內部處理單獨的組別
            if grad_norm > self.grad_norm_clip:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.detach().mul_(self.grad_norm_clip / grad_norm)

            # === 平滑 ratio 變化 ===
            self.grad_ratio_smooth = 0.9 * self.grad_ratio_smooth + 0.1 * ratio

            desired_wd = wd_base * self.grad_ratio_smooth * loss_factor
            desired_wd = max(self.min_wd, min(self.max_wd, desired_wd))

            # === 限制變動幅度 ±30% ===
            lower, upper = old_wd_ema * (1 - self.wd_change_limit), old_wd_ema * (1 + self.wd_change_limit)
            desired_wd = max(lower, min(upper, desired_wd))

            new_wd_ema = self.ema_decay * old_wd_ema + (1 - self.ema_decay) * desired_wd
            group["wd_ema"] = new_wd_ema
            group["weight_decay"] = new_wd_ema

            # ----------------------------------------------------
            # *** 此處為修改點：移除原始的 LR 動態調整程式碼 ***
            # ----------------------------------------------------


# ======================================================
# Optimizer Builder
# ======================================================
def get_optimizer(model, config, optimizer_type="adamw_adaptive_dynamicwd"):
    lr = getattr(config.model, "learning_rate", 1e-4)
    wd = getattr(config.model, "weight_decay", 0.0)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = n.lower()
        if lname.endswith(".bias") or "norm" in lname:
            no_decay.append(p)
        else:
            decay.append(p)
    if optimizer_type == "adamw_adaptive_dynamicwd":
        # *** OPTIMIZER_TYPE 判斷不變 ***
        opt = AdamW_Adaptive_DynamicWD([ 
            {"params": decay, "weight_decay": wd, "lr": lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": lr}
        ], lr=lr, weight_decay=wd)
        logging.info(f"✅ Using AdamW_Adaptive_DynamicWD (WD Only) | lr={lr:.1e} wd={wd:.1e}")
        return opt
    else:
        logging.info(f"✅ Using standard AdamW | lr={lr:.1e} wd={wd:.1e}")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


# ======================================================
# Scheduler (Warmup + Hold + Cosine)
# ======================================================
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-7, hold_epochs=10, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.hold_epochs = hold_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            return [b * (e + 1) / max(1, self.warmup_epochs) for b in self.base_lrs]
        elif e < self.warmup_epochs + self.hold_epochs:
            return self.base_lrs
        p = (e - self.warmup_epochs - self.hold_epochs) / float(max(1, self.max_epochs - self.warmup_epochs - self.hold_epochs))
        return [self.min_lr + (b - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * p)) for b in self.base_lrs]


# ======================================================
# Save / Load
# ======================================================
def save_checkpoint(path, model, optimizer, scheduler, best_loss, loss_history, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
        "loss_history": loss_history,
        "epoch": epoch,
        "loss_weights": LOSS_WEIGHTS
    }, path)
    logging.info(f"[💾 save] checkpoint saved at {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    global LOSS_WEIGHTS
    if not os.path.exists(path):
        logging.warning(f"[resume] No checkpoint found: {path}")
        return 0, float("inf"), []
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and "scheduler" in state and state["scheduler"]:
        scheduler.load_state_dict(state["scheduler"])
    if "loss_weights" in state:
        LOSS_WEIGHTS.update(state["loss_weights"])
    best_loss = state.get("best_loss", float("inf"))
    loss_history = state.get("loss_history", [])
    start_epoch = state.get("epoch", 0) + 1
    logging.info(f"✅ Resume success | epoch={start_epoch} | best_loss={best_loss:.6f}")
    return start_epoch, best_loss, loss_history


# ======================================================
# Trainer
# ======================================================
def trainer(model, category, config):
    device = config.model.device
    model.to(device)
    optimizer = get_optimizer(model, config, OPTIMIZER_TYPE)

    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, config.model.epochs, hold_epochs=10) if USE_WARMUP else None
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=REDUCE_FACTOR, patience=REDUCE_PATIENCE, min_lr=MIN_LR
    ) if USE_REDUCEONPLATEAU else None

    base_dir = os.path.join(os.getcwd(), config.model.checkpoint_dir, category)
    os.makedirs(base_dir, exist_ok=True)

    try:
        reset_adaptive_state()
    except:
        pass

    start_epoch, best_loss, loss_history = 0, float("inf"), []
    latest_ckpt = get_latest_checkpoint(base_dir)
    if RESUME and latest_ckpt:
        start_epoch, best_loss, loss_history = load_checkpoint(latest_ckpt, model, optimizer, scheduler, device)

    train_dataset = Dataset_maker(root=config.data.data_dir, category=category, config=config, is_train=True)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.data.batch_size,
                                         shuffle=True, num_workers=config.model.num_workers, drop_last=True)

    scaler = torch.cuda.amp.GradScaler()
    total_epochs = int(config.model.epochs)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        e_loss, e_raw, e_feats, e_combine, n = 0.0, 0.0, [], 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch [{epoch+1}/{total_epochs}]")
        for imgs, *_ in pbar:
            imgs = imgs.to(device)
            t = torch.randint(0, config.model.trajectory_steps, (imgs.shape[0],), device=device).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss, detail = get_loss(model, imgs, t, config, optimizer=optimizer)
                loss = torch.nan_to_num(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            e_loss += loss.item()
            e_raw += detail["raw"]
            e_combine += detail["combine"]
            if not e_feats:
                e_feats = [0.0]*len(detail["features"])
            for i,v in enumerate(detail["features"]):
                e_feats[i]+=v
            n+=1

        avg_loss = e_loss/max(1,n)
        avg_raw = e_raw/max(1,n)
        avg_feats = [v/max(1,n) for v in e_feats]
        avg_combine = e_combine/max(1,n)
        loss_history.append(avg_loss)

        print(f"\n[Epoch {epoch+1}] Loss={avg_loss:.6f} | Raw={avg_raw:.6f} | Feat={avg_feats} | Combine={avg_combine:.6f}")
        print_adaptive_weights()
        print(f"[Loss Weights] {LOSS_WEIGHTS}")

        if isinstance(optimizer, AdamW_Adaptive_DynamicWD):
            optimizer.step_dynamic_wd(epoch, avg_loss)

        # 🔹 Scheduler: Cosine → Plateau 順序 (外部 LR 調整機制獨立運作)
        if scheduler:
            scheduler.step()
        if reduce_on_plateau and (epoch + 1) % 3 == 0:
            reduce_on_plateau.step(avg_loss)

        # === 儲存 ===
        if (epoch + 1) % SAVE_EVERY_EPOCH == 0:
            save_checkpoint(os.path.join(base_dir, f"{epoch+1}"), model, optimizer, scheduler, best_loss, loss_history, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(os.path.join(base_dir, "best"), model, optimizer, scheduler, best_loss, loss_history, epoch)
            logging.info(f"✅ New best loss: {best_loss:.6f}")

    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.title(f"Training Loss - {category}")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True)
    fig_path = os.path.join(base_dir, f"{category}_LOSS.png")
    plt.savefig(fig_path); plt.close()
    print(f"📉 Loss curve saved at {fig_path}")
    print("Training completed.")
    return model