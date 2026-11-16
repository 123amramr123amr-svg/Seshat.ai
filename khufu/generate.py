"""Generate an image with Khufu from the command line.

Supports high-resolution output (up to 750x750) via super-resolution.

Examples:
    python3 generate.py --subject cat --size 128 --target-size 750 --steps 100 --seed 7 --out cat.png
    python3 generate.py --subject "ارسم قطة" --target-size 750 --out out.png
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

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


def upscale_image(img, target_size, method="bicubic"):
    """Upscale image to target size with super-resolution.
    
    Args:
        img: PIL Image
        target_size: target width/height
        method: "bicubic", "lanczos", or "cascade"
    
    Returns:
        upscaled PIL Image
    """
    if img.size[0] == target_size:
        return img
    
    if method == "cascade":
        # Cascade upscaling: 128 -> 256 -> 512 -> 750
        current = img
        sizes = []
        s = img.size[0]
        while s < target_size:
            s = min(s * 2, target_size)
            sizes.append(s)
        
        for s in sizes:
            current = current.resize((s, s), Image.BICUBIC)
            # Apply mild sharpening after each upscale
            if s < target_size:
                current = current.filter(ImageFilter.SHARPEN)
        
        return current
    else:
        # Single-step upscaling
        resample = Image.BICUBIC if method == "bicubic" else Image.LANCZOS
        return img.resize((target_size, target_size), resample)


def enhance_image(img, sharpness=1.5, contrast=1.1, saturation=1.1):
    """Enhance image quality after upscaling.
    
    Args:
        img: PIL Image
        sharpness: sharpening factor (1.0 = no change, >1.0 = sharper)
        contrast: contrast factor
        saturation: saturation factor
    
    Returns:
        enhanced PIL Image
    """
    # Sharpening
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    
    # Contrast
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    
    # Saturation
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    
    return img


def generate(subject, size=128, steps=50, seed=None, ckpt=None, 
             target_size=None, enhance=True):
    """Generate an image.
    
    Args:
        subject: class name or description
        size: generation size (32, 64, 96, 128)
        steps: DDIM steps (20, 50, 100)
        seed: random seed
        ckpt: checkpoint path
        target_size: output size (if different from size, will upscale)
        enhance: apply enhancement after upscaling
    
    Returns:
        (PIL Image, class_id)
    """
    cid = subject if subject in scenes.CLASS_IDS else scenes.class_from_text(subject)
    if cid is None:
        raise ValueError("مش عارف أرسم «" + str(subject) + "» — المتاح: "
                         + "، ".join(scenes.ARABIC.values()))
    
    model, _cfg = load_model(ckpt)
    y = torch.tensor([scenes.CLASS_IDS.index(cid)])
    
    with torch.no_grad():
        out = sample_ddim(model, y, size=size, steps=steps, seed=seed)
    
    img_array = (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Upscale if needed
    if target_size and target_size != size:
        # Use cascade upscaling for large targets
        if target_size > size * 4:
            img = upscale_image(img, target_size, method="cascade")
        else:
            img = upscale_image(img, target_size, method="bicubic")
        
        # Enhance
        if enhance:
            img = enhance_image(img, sharpness=1.5, contrast=1.1, saturation=1.1)
    
    return img, cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--size", type=int, default=128, choices=[32, 64, 96, 128],
                    help="Generation size (default: 128)")
    ap.add_argument("--target-size", type=int, default=None,
                    help="Output size (if different, will upscale with super-resolution)")
    ap.add_argument("--steps", type=int, default=50, choices=[20, 50, 100])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--ckpt", default=None, help="Path to checkpoint")
    ap.add_argument("--no-enhance", action="store_true", help="Disable enhancement")
    a = ap.parse_args()
    
    img, cid = generate(a.subject, size=a.size, steps=a.steps, seed=a.seed, 
                        ckpt=a.ckpt, target_size=a.target_size, 
                        enhance=not a.no_enhance)
    img.save(a.out)
    
    out_size = img.size[0]
    print(f"{cid} -> {a.out} ({out_size}x{out_size}, {a.steps} steps)")


if __name__ == "__main__":
    main()
