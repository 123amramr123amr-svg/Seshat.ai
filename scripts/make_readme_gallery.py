# -*- coding: utf-8 -*-
"""توليد صور معرض الـ README بجولة خ5 (الرسم الهجين) — صور خوفو الحقيقية."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "khufu"))

import numpy as np
import png
import generate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "khufu")
os.makedirs(OUT, exist_ok=True)

# (الصنف، البذرة، الاسم بالملف)
GALLERY = [
    ("boat", 7, "boat"),
    ("balloons", 11, "balloons"),
    ("birds", 5, "birds"),
    ("road", 3, "road"),
    ("sunset", 42, "sunset"),
    ("mountains", 9, "mountains"),
    ("city", 21, "city"),
    ("sea", 5, "sea"),
]

paths = []
for cid, seed, name in GALLERY:
    img, got = generate.generate(cid, size=750, steps=50, seed=seed)
    p = os.path.join(OUT, "%s_750.png" % name)
    img.save(p)
    paths.append((name, img.arr))
    print("OK %s -> %s" % (name, p), flush=True)

# كولاج 4×2 من الصور نفسها (مصغّرة 360) لعرض سريع في الـ README
from resize import resize_lanczos
CELL = 360
grid = np.zeros((CELL * 2, CELL * 4, 3), dtype=np.uint8)
for i, (name, arr) in enumerate(paths[:8]):
    small = resize_lanczos(arr.astype(np.float32) / 255.0, CELL)
    r, c = divmod(i, 4)
    grid[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = \
        np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)
gp = os.path.join(OUT, "grid_khufu_1536.png")
with open(gp, "wb") as f:
    f.write(png.encode_png(grid))
print("OK grid ->", gp, flush=True)
