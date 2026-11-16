"""Train Khufu (diffusion model) on procedurally rendered scenes.

Usage:
    python3 train_khufu.py --steps 60000 --minutes 120 --batch 8 --base 32
    python3 train_khufu.py --resume --minutes 120   # continue from data/khufu_latest.pt
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import scenes
from khufu_model import EMA, UNet, get_schedule, sample_ddim

SAMPLE_DIR = "data/samples"
STATUS = "data/status.json"


def save_samples(ema, cfg, step, steps=30):
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


def save_ckpt(path, model, ema, opt, step, cfg):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "config": cfg}, tmp)
    os.replace(tmp, path)


def write_status(step, loss, elapsed_min, sample):
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump({"step": step, "loss": loss, "elapsed_min": round(elapsed_min, 1),
                   "sample": sample}, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--minutes", type=float, default=0,
                    help="stop after this many minutes (0 = no limit)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="data/khufu_latest.pt")
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    rng = random.Random(a.seed)
    _, _, acp = get_schedule()
    cfg = {"base": a.base, "blocks": a.blocks, "n_classes": len(scenes.CLASSES), "size": a.size}
    model = UNet(base=a.base, n_classes=len(scenes.CLASSES), blocks=a.blocks)
    ema = EMA(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    step = 0
    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        cfg = ck.get("config", cfg)
        print(f"resumed at step {step}", flush=True)

    print(f"khufu: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params, "
          f"batch {a.batch}, size {a.size}, target {a.steps} steps", flush=True)
    t0 = time.time()
    loss_acc, loss_n, last_loss = 0.0, 0, 0.0
    while step < a.steps:
        if a.minutes and (time.time() - t0) / 60 > a.minutes:
            break
        xs, ys = [], []
        for _ in range(a.batch):
            cid = rng.choice(scenes.CLASS_IDS)
            img = scenes.render(cid, rng)
            if rng.random() < 0.5:
                img = img[:, ::-1].copy()
            xs.append(img)
            ys.append(scenes.CLASS_IDS.index(cid))
        x = torch.from_numpy(np.stack(xs)).float().permute(0, 3, 1, 2) / 127.5 - 1
        y = torch.tensor(ys)
        t = torch.randint(0, 1000, (a.batch,))
        noise = torch.randn_like(x)
        acp_t = acp[t][:, None, None, None]
        x_noisy = acp_t.sqrt() * x + (1 - acp_t).sqrt() * noise
        pred = model(x_noisy, t, y)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        loss_acc += loss.item()
        loss_n += 1
        step += 1
        if step % 50 == 0:
            last_loss = loss_acc / max(1, loss_n)
            print(f"step {step} loss {last_loss:.4f} elapsed {(time.time()-t0)/60:.1f}min",
                  flush=True)
            loss_acc, loss_n = 0.0, 0
        if step % a.sample_every == 0:
            sample = save_samples(ema, cfg, step)
            save_ckpt(a.out, model, ema, opt, step, cfg)
            write_status(step, last_loss, (time.time() - t0) / 60, sample)
            print(f"  saved samples + checkpoint at step {step}", flush=True)
    sample = save_samples(ema, cfg, step, steps=50)
    save_ckpt(a.out, model, ema, opt, step, cfg)
    write_status(step, last_loss, (time.time() - t0) / 60, sample)
    print(f"done at step {step}, checkpoint {a.out}", flush=True)


if __name__ == "__main__":
    main()
