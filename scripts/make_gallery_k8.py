# -*- coding: utf-8 -*-
"""عينات جولة خ8 — قوس قزح + طبقة الطاقة على 750×750، مع تحقق رقمي صارم.
كل صورة لازم تكون: ملف >20KB و std >30 (مش صورة فاضية أو باهتة).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "khufu"))

import numpy as np
import png
import generate
from resize import resize_lanczos

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "khufu")
os.makedirs(OUT, exist_ok=True)

# (النص للتركيبة أو الصنف، البذرة، اسم الملف، هل نص تركيبة)
JOBS = [
    ("rainbow", 14, "rainbow_750", False),
    ("قارب في البحر وقت الغروب", 7, "boat_sunset_750", True),
    ("طيور فوق الجبال مع قوس قزح", 21, "birds_mountains_rainbow_750", True),
    ("قوس قزح فوق البحر", 5, "rainbow_sea_750", True),
]

results = []
arrs = {}
for text, seed, name, composite in JOBS:
    if composite:
        img, got = generate.generate_composite(text, size=750, steps=50, seed=seed)
    else:
        img, got = generate.generate(text, size=750, steps=50, seed=seed)
    p = os.path.join(OUT, "%s.png" % name)
    img.save(p)
    data = open(p, "rb").read()
    std = float(np.std(img.arr))
    ok = len(data) > 20_000 and std > 30.0
    results.append((name, got, len(data), std, ok))
    arrs[name] = img.arr.astype(np.float32) / 255.0
    print("%s | فهم: %-22s | %6d bytes | std %5.1f | %s"
          % (name, got, len(data), std, "OK" if ok else "FAIL"), flush=True)

# شبكة خ8: 8 مشاهد (منهم قوس قزح والتركيبات الجديدة) صفين × 4 أعمدة على 360
CELL = 360
grid = np.zeros((CELL * 2, CELL * 4, 3), dtype=np.uint8)
for i, (text, seed, name, composite) in enumerate(JOBS[:4]):
    small = resize_lanczos(arrs[name], CELL)
    grid[0:CELL, i * CELL:(i + 1) * CELL] = \
        np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)
# 4 أصناف من الأصناف العادية للتنوع (الصف التاني)
for i, (cid, seed) in enumerate([("boat", 7), ("city", 21), ("sunset", 42), ("birds", 5)]):
    img, _ = generate.generate(cid, size=750, steps=50, seed=seed)
    small = resize_lanczos(img.arr.astype(np.float32) / 255.0, CELL)
    grid[CELL:CELL * 2, i * CELL:(i + 1) * CELL] = \
        np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)
gp = os.path.join(OUT, "grid_kh8_1536.png")
with open(gp, "wb") as f:
    f.write(png.encode_png(grid))
print("OK grid ->", gp, len(open(gp, "rb").read()), "bytes", flush=True)

fails = [r for r in results if not r[4]]
print("--- النتيجة: %d/%d عدت ---" % (len(results) - len(fails), len(results)))
sys.exit(1 if fails else 0)
