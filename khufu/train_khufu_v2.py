"""Train Khufu v2 (enhanced diffusion model) with improvements.

Improvements over v1:
- Data augmentation (horizontal flip, random crop, color jitter)
- Learning rate scheduling (cosine annealing)
- Gradient clipping for stability
- Larger model support (base=48 or 64)
- Better evaluation (more DDIM steps)

Usage:
    python3 train_khufu_v2.py --steps 50000 --minutes 180 --batch 8 --base 48
    python3 train_khufu_v2.py --resume --minutes 180
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

import scenes
from khufu_model import EMA, UNet, get_schedule, sample_ddim

SAMPLE_DIR = "data/samples"
STATUS = "data/status.json"


def augment_image(img_array, rng):
    """Apply data augmentation to training images.
    
    Args:
        img_array: numpy array (H, W, 3) uint8
        rng: random.Random instance
    
    Returns:
        augmented numpy array
    """
    img = Image.fromarray(img_array)
    
    # Horizontal flip (50% chance)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random crop (10% margin)
    if rng.random() < 0.3:
        w, h = img.size
        margin = int(min(w, h) * 0.1)
        if margin > 0:
            left = rng.randint(0, margin)
            top = rng.randint(0, margin)
            right = w - rng.randint(0, margin)
            bottom = h - rng.randint(0, margin)
            img = img.crop((left, top, right, bottom))
            img = img.resize((w, h), Image.BILINEAR)
    
    # Color jitter (brightness, contrast)
    if rng.random() < 0.2:
        brightness = rng.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(brightness)
    
    if rng.random() < 0.2:
        contrast = rng.uniform(0.8, 1.2)
        img = ImageEnhance.Contrast(img).enhance(contrast)
    
    return np.array(img)


def cosine_lr_schedule(optimizer, step, total_steps, base_lr, min_lr=1e-6):
    """Cosine annealing learning rate schedule.
    
    Args:
        optimizer: PyTorch optimizer
        step: current step
        total_steps: total training steps
        base_lr: base learning rate
        min_lr: minimum learning rate
    """
    progress = step / total_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine_decay
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def save_samples(ema, cfg, step, steps=50):
    """Generate and save sample images."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    model = UNet(base=cfg["base"], n_classes=len(scenes.CLASSES), blocks=cfg["blocks"])
    ema.apply_to(model)
    model.eval()
    y = torch.arange(len(scenes.CLASSES))
    with torch.no_grad():
        out = sample_ddim(model, y, size=64, steps=steps, seed=1000)
    imgs = (out.permute(0, 2, 3, 1).clamp(0, 1).numpy() * 255).astype(np.uint8)
    cols, cell, label_h = 5, 64, 14
    rows = (len(imgs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (20, 20, 24))
    d = ImageDraw.Draw(grid)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (cid, _ar, _k) in enumerate(scenes.CLASSES):
        x0, y0 = (i % cols) * cell, (i // cols) * (cell + label_h)
        grid.paste(Image.fromarray(imgs[i]), (x0, y0))
        d.text((x0 + 3, y0 + cell + 2), cid, fill=(220, 220, 220), font=font)
    path = os.path.join(SAMPLE_DIR, f"step_{step:06d}.png")
    grid.save(path)
    grid.save(os.path.join(SAMPLE_DIR, "latest.png"))
    return path


def save_ckpt(path, model, ema, opt, step, cfg, lr):
    """Save checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "config": cfg, "lr": lr}, tmp)
    os.replace(tmp, path)


def write_status(step, loss, elapsed_min, sample, lr):
    """Write training status to JSON."""
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump({"step": step, "loss": loss, "elapsed_min": round(elapsed_min, 1),
                   "sample": sample, "lr": lr}, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000, help="Total training steps")
    ap.add_argument("--minutes", type=float, default=0, help="Stop after this many minutes (0 = no limit)")
    ap.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    ap.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--base", type=int, default=48, help="Base channels (32=1.96M, 48=4.4M, 64=7.8M)")
    ap.add_argument("--blocks", type=int, default=2, help="Residual blocks per level")
    ap.add_argument("--size", type=int, default=64, help="Training image size")
    ap.add_argument("--threads", type=int, default=2, help="CPU threads")
    ap.add_argument("--out", default="data/khufu_latest.pt", help="Checkpoint path")
    ap.add_argument("--sample-every", type=int, default=1000, help="Save samples every N steps")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    ap.add_argument("--seed", type=int, default=7, help="Random seed")
    ap.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    rng = random.Random(a.seed)
    _, _, acp = get_schedule()
    cfg = {"base": a.base, "blocks": a.blocks, "n_classes": len(scenes.CLASSES), "size": a.size}
    
    model = UNet(base=a.base, n_classes=len(scenes.CLASSES), blocks=a.blocks)
    ema = EMA(model, decay=0.9999)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    
    step = 0
    current_lr = a.lr
    
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        cfg = ck.get("config", cfg)
        current_lr = ck.get("lr", a.lr)
        print(f"✓ Resumed at step {step}, lr={current_lr:.6f}", flush=True)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Khufu v2: {num_params/1e6:.2f}M params, batch {a.batch}, size {a.size}, base {a.base}", flush=True)
    print(f"Target: {a.steps} steps, lr {a.lr} → {a.min_lr} (cosine), grad clip {a.grad_clip}", flush=True)
    
    t0 = time.time()
    loss_acc, loss_n, last_loss = 0.0, 0, 0.0
    
    while step < a.steps:
        if a.minutes and (time.time() - t0) / 60 > a.minutes:
            print(f"Time limit reached ({a.minutes} min)", flush=True)
            break
        
        # Learning rate scheduling
        current_lr = cosine_lr_schedule(opt, step, a.steps, a.lr, a.min_lr)
        
        # Generate batch with augmentation
        xs, ys = [], []
        for _ in range(a.batch):
            cid = rng.choice(scenes.CLASS_IDS)
            img = scenes.render(cid, rng)
            
            # Data augmentation
            img = augment_image(img, rng)
            
            xs.append(img)
            ys.append(scenes.CLASS_IDS.index(cid))
        
        x = torch.from_numpy(np.stack(xs)).float().permute(0, 3, 1, 2) / 127.5 - 1
        y = torch.tensor(ys)
        t = torch.randint(0, 1000, (a.batch,))
        noise = torch.randn_like(x)
        acp_t = acp[t][:, None, None, None]
        x_noisy = acp_t.sqrt() * x + (1 - acp_t).sqrt() * noise
        
        # Forward pass
        pred = model(x_noisy, t, y)
        loss = F.mse_loss(pred, noise)
        
        # Backward pass with gradient clipping
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        
        # Update EMA
        ema.update(model)
        
        loss_acc += loss.item()
        loss_n += 1
        step += 1
        
        # Logging
        if step % 50 == 0:
            last_loss = loss_acc / max(1, loss_n)
            elapsed = (time.time() - t0) / 60
            print(f"Step {step:6d} | Loss {last_loss:.5f} | LR {current_lr:.6f} | {elapsed:.1f}min", flush=True)
            loss_acc, loss_n = 0.0, 0
        
        # Save samples and checkpoint
        if step % a.sample_every == 0:
            sample = save_samples(ema, cfg, step, steps=50)
            save_ckpt(a.out, model, ema, opt, step, cfg, current_lr)
            write_status(step, last_loss, (time.time() - t0) / 60, sample, current_lr)
            print(f"  ✓ Saved checkpoint + samples at step {step}", flush=True)
    
    # Final save
    sample = save_samples(ema, cfg, step, steps=50)
    save_ckpt(a.out, model, ema, opt, step, cfg, current_lr)
    write_status(step, last_loss, (time.time() - t0) / 60, sample, current_lr)
    print(f"✓ Training complete at step {step}, checkpoint {a.out}", flush=True)


if __name__ == "__main__":
    main()
