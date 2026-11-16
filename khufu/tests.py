"""Quick tests for Khufu (no training). Run: python3 tests.py"""
import random

import numpy as np
import torch

import scenes
from khufu_model import UNet, sample_ddim


def test_scenes():
    rng = random.Random(0)
    for cid in scenes.CLASS_IDS:
        img = scenes.render(cid, rng)
        assert img.shape == (64, 64, 3), (cid, img.shape)
        assert img.dtype == np.uint8
        assert img.std() > 5, f"{cid} looks blank"
    print("scenes ok")


def test_keywords():
    assert scenes.class_from_text("ارسم قطة") == "cat"
    assert scenes.class_from_text("pyramid") == "pyramid"
    assert scenes.class_from_text("عربية حمرا") == "car"
    assert scenes.class_from_text("xyz") is None
    assert scenes.class_from_text("") is None
    print("keywords ok")


def test_model():
    m = UNet(base=16, blocks=1)
    m.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = m(x, torch.tensor([10, 20]), torch.tensor([0, 1]))
    assert out.shape == x.shape
    for s in (32, 96, 128):
        x = torch.randn(1, 3, s, s)
        with torch.no_grad():
            out = m(x, torch.tensor([5]), torch.tensor([2]))
        assert out.shape == x.shape, s
    print("model ok")


def test_sample():
    m = UNet(base=16, blocks=1)
    m.eval()
    out = sample_ddim(m, torch.tensor([3]), size=32, steps=5, seed=1)
    assert out.shape == (1, 3, 32, 32)
    assert float(out.min()) >= 0 and float(out.max()) <= 1
    out2 = sample_ddim(m, torch.tensor([3]), size=32, steps=5, seed=1)
    assert torch.equal(out, out2), "same seed should give same image"
    print("sample ok")


if __name__ == "__main__":
    test_scenes()
    test_keywords()
    test_model()
    test_sample()
    print("✅ كل الاختبارات عدت")
