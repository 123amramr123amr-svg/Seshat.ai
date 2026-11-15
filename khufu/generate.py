"""Generate an image with Khufu from the command line.

Examples:
    python3 generate.py --subject cat --size 128 --steps 100 --seed 7 --out cat.png
    python3 generate.py --subject "ارسم قطة" --out out.png
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

import scenes
from khufu_model import UNet, sample_ddim

# Use absolute paths relative to this file's directory
KHUFU_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(KHUFU_DIR, "data", "khufu_latest.pt")


def load_model(path=None):
    if path is None:
        path = DEFAULT_CKPT
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = UNet(base=cfg.get("base", 32),
                 n_classes=cfg.get("n_classes", len(scenes.CLASSES)),
                 blocks=cfg.get("blocks", 1))
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, cfg


def generate(subject, size=64, steps=50, seed=None, ckpt=None):
    cid = subject if subject in scenes.CLASS_IDS else scenes.class_from_text(subject)
    if cid is None:
        raise ValueError("مش عارف أرسم «" + str(subject) + "» — المتاح: "
                         + "، ".join(scenes.ARABIC.values()))
    model, _cfg = load_model(ckpt)
    y = torch.tensor([scenes.CLASS_IDS.index(cid)])
    with torch.no_grad():
        out = sample_ddim(model, y, size=size, steps=steps, seed=seed)
    img = (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(img), cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--size", type=int, default=64, choices=[32, 64, 96, 128])
    ap.add_argument("--steps", type=int, default=50, choices=[20, 50, 100])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint (default: data/khufu_latest.pt)")
    a = ap.parse_args()
    img, cid = generate(a.subject, size=a.size, steps=a.steps, seed=a.seed, ckpt=a.ckpt)
    img.save(a.out)
    print(f"{cid} -> {a.out} ({a.size}x{a.size}, {a.steps} steps)")


if __name__ == "__main__":
    main()
