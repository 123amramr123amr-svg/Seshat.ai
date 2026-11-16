"""Procedural scene renderer for Khufu.

Khufu is trained on scenes rendered by this file — no downloads, no external data.
Each class has a recipe: a background + a main subject + optional extras.
Rendering happens at 128x128 (2x supersampling) and is downscaled to 64x64.
"""
import math
import random

import numpy as np
from PIL import Image, ImageDraw

W = 128   # render canvas
OUT = 64  # training size

CLASSES = [
    ("sun",      "شمس",   ["شمس", "sun"]),
    ("moon",     "قمر",   ["قمر", "هلال", "moon"]),
    ("star",     "نجمة",  ["نجمة", "نجوم", "star"]),
    ("cloud",    "سحابة", ["سحابة", "سحاب", "غيوم", "cloud"]),
    ("bird",     "طائر",  ["طائر", "طيور", "عصفور", "bird"]),
    ("tree",     "شجرة",  ["شجرة", "شجر", "tree"]),
    ("palm",     "نخلة",  ["نخلة", "نخل", "palm"]),
    ("house",    "بيت",   ["بيت", "منزل", "house"]),
    ("car",      "عربية", ["عربية", "عربيه", "سيارة", "car"]),
    ("cat",      "قطة",   ["قطة", "قطه", "cat"]),
    ("dog",      "كلب",   ["كلب", "dog"]),
    ("horse",    "حصان",  ["حصان", "horse"]),
    ("camel",    "جمل",   ["جمل", "camel"]),
    ("pyramid",  "هرم",   ["هرم", "أهرامات", "اهرامات", "pyramid"]),
    ("mountain", "جبل",   ["جبل", "جبال", "mountain"]),
    ("sea",      "بحر",   ["بحر", "sea", "ocean"]),
    ("boat",     "مركب",  ["مركب", "سفينة", "فلوكة", "boat"]),
    ("fish",     "سمكة",  ["سمكة", "سمك", "fish"]),
    ("flower",   "وردة",  ["وردة", "ورد", "زهرة", "flower"]),
]
CLASS_IDS = [c[0] for c in CLASSES]
ARABIC = {c[0]: c[1] for c in CLASSES}


def class_from_text(text):
    """Map free text (Arabic/English) to a class id, or None."""
    t = (text or "").strip().lower()
    if not t:
        return None
    for cid, _ar, keys in CLASSES:
        for k in keys:
            if k.lower() in t:
                return cid
    return None


def _rr(d, box, radius, **kw):
    fn = getattr(d, "rounded_rectangle", None)
    if fn:
        fn(box, radius=radius, **kw)
    else:
        d.rectangle(box, **kw)


def _gradient(img, top, bottom, y0=0, y1=W):
    h = max(1, y1 - y0)
    col = np.zeros((h, 1, 3), dtype=np.uint8)
    for i in range(h):
        f = i / max(1, h - 1)
        col[i, 0] = [int(top[c] + (bottom[c] - top[c]) * f) for c in range(3)]
    grad = Image.fromarray(col, "RGB").resize((W, h), Image.BILINEAR)
    img.paste(grad, (0, y0))


def _sky(img, mode):
    palettes = {
        "day":    ((58, 123, 213), (207, 232, 255)),
        "sunset": ((64, 24, 110), (255, 140, 90)),
        "night":  ((8, 10, 38), (27, 42, 74)),
        "dawn":   ((90, 110, 180), (255, 190, 140)),
    }
    top, bottom = palettes[mode]
    _gradient(img, top, bottom)


def _ground(img, top, bottom, frac=0.22):
    _gradient(img, top, bottom, int(W * (1 - frac)), W)


GRASS = ((96, 170, 80), (52, 110, 44))
SAND = ((238, 214, 150), (214, 178, 110))
ROAD = ((112, 112, 118), (78, 78, 84))
WATER = ((36, 116, 186), (10, 60, 120))


def _sun(d, cx, cy, r, color=(255, 205, 70)):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    for a in range(0, 360, 45):
        x1 = cx + (r + 2) * math.cos(math.radians(a))
        y1 = cy + (r + 2) * math.sin(math.radians(a))
        x2 = cx + (r + 8) * math.cos(math.radians(a))
        y2 = cy + (r + 8) * math.sin(math.radians(a))
        d.line([x1, y1, x2, y2], fill=color, width=3)


def _moon(d, cx, cy, r):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(242, 240, 214))
    for dx, dy, cr in [(-0.3, -0.2, 0.22), (0.25, 0.15, 0.18), (0.05, -0.45, 0.14)]:
        d.ellipse([cx + (dx - cr) * r, cy + (dy - cr) * r,
                   cx + (dx + cr) * r, cy + (dy + cr) * r], fill=(222, 218, 188))


def _star(d, cx, cy, r, color=(255, 240, 170)):
    pts = [(cx, cy - r), (cx + r * 0.18, cy - r * 0.18), (cx + r, cy),
           (cx + r * 0.18, cy + r * 0.18), (cx, cy + r), (cx - r * 0.18, cy + r * 0.18),
           (cx - r, cy), (cx - r * 0.18, cy - r * 0.18)]
    d.polygon(pts, fill=color)


def _stars(d, rng, n):
    for _ in range(n):
        x, y = rng.uniform(2, W - 2), rng.uniform(2, 70)
        r = rng.uniform(1.5, 3.5)
        if rng.random() < 0.4:
            _star(d, x, y, r * 2.2)
        else:
            d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 250, 220))


def _cloud(d, cx, cy, s, color=(255, 255, 255)):
    for dx, dy, r in [(-0.45, 0.12, 0.3), (0, -0.05, 0.4), (0.45, 0.12, 0.3), (0.1, 0.2, 0.28)]:
        d.ellipse([cx + (dx - r) * s, cy + (dy - r) * s,
                   cx + (dx + r) * s, cy + (dy + r) * s], fill=color)


def _bird(d, cx, cy, s, color=(40, 42, 52)):
    d.arc([cx - s, cy - s * 0.45, cx, cy + s * 0.45], start=180, end=360, fill=color, width=3)
    d.arc([cx, cy - s * 0.45, cx + s, cy + s * 0.45], start=180, end=360, fill=color, width=3)


def _tree(d, cx, base, s, trunk=(122, 82, 52), leaf=(58, 138, 62)):
    d.rectangle([cx - s * 0.06, base - s * 0.5, cx + s * 0.06, base], fill=trunk)
    for dx, dy, r in [(0, -0.78, 0.32), (-0.22, -0.58, 0.23), (0.22, -0.58, 0.23)]:
        d.ellipse([cx + (dx - r) * s, base + (dy - r) * s,
                   cx + (dx + r) * s, base + (dy + r) * s], fill=leaf)


def _palm(d, cx, base, s):
    pts = [(cx, base), (cx + s * 0.07, base - s * 0.35), (cx + s * 0.15, base - s * 0.68)]
    d.line(pts, fill=(140, 100, 60), width=max(2, int(s * 0.07)))
    tx, ty = pts[-1]
    for a in (-170, -135, -100, -65, -30, 5):
        x2 = tx + s * 0.42 * math.cos(math.radians(a))
        y2 = ty + s * 0.3 * math.sin(math.radians(a)) + s * 0.12
        d.line([tx, ty, x2, y2], fill=(48, 128, 58), width=3)
    d.ellipse([tx - 3, ty + 2, tx + 3, ty + 8], fill=(110, 80, 40))


def _house(d, cx, base, s, wall=(242, 232, 212), roof=(182, 72, 58)):
    w, h = s * 0.72, s * 0.46
    d.rectangle([cx - w / 2, base - h, cx + w / 2, base], fill=wall)
    d.polygon([(cx - w / 2 - 4, base - h), (cx + w / 2 + 4, base - h),
               (cx, base - h - s * 0.34)], fill=roof)
    d.rectangle([cx - s * 0.08, base - s * 0.22, cx + s * 0.08, base], fill=(122, 82, 52))
    d.rectangle([cx - s * 0.29, base - h + s * 0.08, cx - s * 0.13, base - h + s * 0.24],
                fill=(150, 200, 230))
    d.rectangle([cx + s * 0.13, base - h + s * 0.08, cx + s * 0.29, base - h + s * 0.24],
                fill=(150, 200, 230))


def _car(d, cx, base, s, color):
    w, h = s * 0.92, s * 0.3
    y = base - h
    _rr(d, [cx - w / 2, y, cx + w / 2, base - s * 0.07], int(s * 0.06), fill=color)
    _rr(d, [cx - w * 0.27, y - s * 0.2, cx + w * 0.27, y + 2], int(s * 0.05), fill=color)
    d.rectangle([cx - w * 0.23, y - s * 0.15, cx - w * 0.02, y - 2], fill=(170, 212, 236))
    d.rectangle([cx + w * 0.02, y - s * 0.15, cx + w * 0.23, y - 2], fill=(170, 212, 236))
    for wx in (cx - w * 0.3, cx + w * 0.3):
        r = s * 0.095
        d.ellipse([wx - r, base - 2 * r, wx + r, base], fill=(28, 28, 34))
        d.ellipse([wx - r * 0.45, base - r * 1.55, wx + r * 0.45, base - r * 0.45],
                  fill=(185, 185, 190))


def _cat(d, cx, base, s, color=(45, 40, 50)):
    d.ellipse([cx - s * 0.22, base - s * 0.42, cx + s * 0.22, base], fill=color)
    d.ellipse([cx - s * 0.16, base - s * 0.72, cx + s * 0.16, base - s * 0.4], fill=color)
    d.polygon([(cx - s * 0.15, base - s * 0.66), (cx - s * 0.04, base - s * 0.66),
               (cx - s * 0.1, base - s * 0.86)], fill=color)
    d.polygon([(cx + s * 0.04, base - s * 0.66), (cx + s * 0.15, base - s * 0.66),
               (cx + s * 0.1, base - s * 0.86)], fill=color)
    d.arc([cx + s * 0.08, base - s * 0.32, cx + s * 0.46, base + s * 0.04],
          start=270, end=90, fill=color, width=max(2, int(s * 0.06)))
    d.ellipse([cx - s * 0.09, base - s * 0.6, cx - s * 0.04, base - s * 0.55],
              fill=(255, 222, 120))
    d.ellipse([cx + s * 0.04, base - s * 0.6, cx + s * 0.09, base - s * 0.55],
              fill=(255, 222, 120))


def _dog(d, cx, base, s, color=(112, 76, 46)):
    d.ellipse([cx - s * 0.3, base - s * 0.46, cx + s * 0.3, base - s * 0.16], fill=color)
    d.ellipse([cx + s * 0.16, base - s * 0.64, cx + s * 0.42, base - s * 0.38], fill=color)
    d.ellipse([cx + s * 0.32, base - s * 0.5, cx + s * 0.48, base - s * 0.38], fill=color)
    d.ellipse([cx + s * 0.14, base - s * 0.64, cx + s * 0.3, base - s * 0.42], fill=(70, 45, 25))
    for lx in (cx - s * 0.2, cx - s * 0.06, cx + s * 0.08, cx + s * 0.2):
        d.rectangle([lx, base - s * 0.2, lx + s * 0.06, base], fill=color)
    d.line([(cx - s * 0.28, base - s * 0.42), (cx - s * 0.44, base - s * 0.56)],
           fill=color, width=max(2, int(s * 0.05)))


def _horse(d, cx, base, s, color=(132, 86, 50)):
    d.ellipse([cx - s * 0.32, base - s * 0.56, cx + s * 0.28, base - s * 0.26], fill=color)
    d.polygon([(cx + s * 0.16, base - s * 0.5), (cx + s * 0.4, base - s * 0.8),
               (cx + s * 0.5, base - s * 0.72), (cx + s * 0.3, base - s * 0.42)], fill=color)
    for lx in (cx - s * 0.24, cx - s * 0.12, cx + s * 0.08, cx + s * 0.2):
        d.line([(lx, base - s * 0.3), (lx, base)], fill=color, width=max(2, int(s * 0.05)))
    d.line([(cx - s * 0.3, base - s * 0.52), (cx - s * 0.44, base - s * 0.3)],
           fill=(60, 40, 25), width=max(2, int(s * 0.06)))
    d.line([(cx + s * 0.18, base - s * 0.54), (cx + s * 0.38, base - s * 0.78)],
           fill=(60, 40, 25), width=max(2, int(s * 0.05)))


def _camel(d, cx, base, s, color=(192, 152, 96)):
    d.ellipse([cx - s * 0.32, base - s * 0.56, cx + s * 0.3, base - s * 0.28], fill=color)
    d.ellipse([cx - s * 0.08, base - s * 0.74, cx + s * 0.14, base - s * 0.46], fill=color)
    d.polygon([(cx + s * 0.2, base - s * 0.5), (cx + s * 0.4, base - s * 0.82),
               (cx + s * 0.48, base - s * 0.76), (cx + s * 0.32, base - s * 0.42)], fill=color)
    for lx in (cx - s * 0.24, cx - s * 0.1, cx + s * 0.1, cx + s * 0.22):
        d.line([(lx, base - s * 0.32), (lx, base)], fill=color, width=max(2, int(s * 0.05)))
    d.line([(cx - s * 0.3, base - s * 0.5), (cx - s * 0.37, base - s * 0.3)], fill=color, width=2)


def _pyramid(d, cx, base, s, color=(222, 192, 122)):
    w, h = s * 0.85, s * 0.62
    d.polygon([(cx - w / 2, base), (cx + w / 2, base), (cx, base - h)], fill=color)
    d.polygon([(cx, base - h), (cx + w / 2, base), (cx, base)], fill=(192, 162, 96))


def _mountain(d, cx, base, s, color=(112, 122, 142)):
    d.polygon([(cx - s * 0.5, base), (cx - s * 0.08, base - s * 0.66), (cx + s * 0.34, base)],
              fill=color)
    d.polygon([(cx - s * 0.08, base - s * 0.66), (cx - s * 0.02, base - s * 0.52),
               (cx + s * 0.04, base - s * 0.66)], fill=(242, 246, 250))
    d.polygon([(cx - s * 0.02, base), (cx + s * 0.3, base - s * 0.46), (cx + s * 0.56, base)],
              fill=(96, 106, 126))
    d.polygon([(cx + s * 0.3, base - s * 0.46), (cx + s * 0.34, base - s * 0.38),
               (cx + s * 0.38, base - s * 0.46)], fill=(242, 246, 250))


def _boat(d, cx, water_y, s, hull=(152, 82, 52), sail=(246, 246, 240)):
    w = s * 0.72
    d.polygon([(cx - w / 2, water_y), (cx + w / 2, water_y),
               (cx + w * 0.35, water_y + s * 0.18), (cx - w * 0.35, water_y + s * 0.18)],
              fill=hull)
    d.line([(cx, water_y), (cx, water_y - s * 0.55)], fill=(102, 72, 42), width=3)
    d.polygon([(cx + 3, water_y - s * 0.5), (cx + 3, water_y - 2), (cx + s * 0.34, water_y - 2)],
              fill=sail)
    d.polygon([(cx - 3, water_y - s * 0.4), (cx - 3, water_y - 2), (cx - s * 0.27, water_y - 2)],
              fill=(232, 122, 92))


def _fish(d, cx, cy, s, color=(240, 140, 60)):
    d.ellipse([cx - s * 0.3, cy - s * 0.16, cx + s * 0.2, cy + s * 0.16], fill=color)
    d.polygon([(cx - s * 0.3, cy), (cx - s * 0.5, cy - s * 0.17), (cx - s * 0.5, cy + s * 0.17)],
              fill=color)
    d.ellipse([cx + s * 0.07, cy - s * 0.05, cx + s * 0.14, cy + s * 0.02], fill=(20, 20, 26))


def _flower(d, cx, base, s, petal=(232, 92, 142)):
    d.line([(cx, base), (cx, base - s * 0.42)], fill=(58, 138, 62), width=3)
    cy = base - s * 0.52
    for a in range(0, 360, 72):
        x = cx + s * 0.13 * math.cos(math.radians(a))
        y = cy + s * 0.13 * math.sin(math.radians(a))
        d.ellipse([x - s * 0.09, y - s * 0.09, x + s * 0.09, y + s * 0.09], fill=petal)
    d.ellipse([cx - s * 0.07, cy - s * 0.07, cx + s * 0.07, cy + s * 0.07], fill=(255, 212, 82))


def _waves(d, y, color=(205, 232, 252), amp=3, width=3):
    pts = [(x, y + amp * math.sin(x / 9.0)) for x in range(0, W, 4)]
    d.line(pts, fill=color, width=width)


def render(cid, rng=None):
    """Render one scene of class cid. Returns uint8 array (64, 64, 3)."""
    rng = rng or random.Random()
    img = Image.new("RGB", (W, W))
    d = ImageDraw.Draw(img)
    j = lambda a: rng.uniform(-a, a)
    day = rng.random() < 0.6

    if cid == "sun":
        _sky(img, "day" if day else "sunset")
        _ground(img, *GRASS)
        _sun(d, 64 + j(20), 34 + j(8), rng.uniform(11, 15))
        _cloud(d, 30 + j(10), 26 + j(6), rng.uniform(26, 36))
        _cloud(d, 96 + j(10), 20 + j(6), rng.uniform(22, 30))
    elif cid == "moon":
        _sky(img, "night")
        _ground(img, (20, 26, 44), (10, 14, 26))
        _stars(d, rng, 26)
        _moon(d, 84 + j(14), 30 + j(8), rng.uniform(11, 14))
    elif cid == "star":
        _sky(img, "night")
        _ground(img, (20, 26, 44), (10, 14, 26))
        _stars(d, rng, 14)
        _star(d, 64 + j(16), 44 + j(10), rng.uniform(13, 18))
        _star(d, 34 + j(8), 26 + j(6), rng.uniform(7, 10))
        _star(d, 96 + j(8), 24 + j(6), rng.uniform(7, 10))
    elif cid == "cloud":
        _sky(img, "day" if day else "dawn")
        _cloud(d, 64 + j(10), 52 + j(8), rng.uniform(44, 56))
        _cloud(d, 34 + j(8), 38 + j(6), rng.uniform(26, 34))
        _cloud(d, 96 + j(8), 34 + j(6), rng.uniform(24, 32))
        if day:
            _sun(d, 104, 18, 9)
    elif cid == "bird":
        _sky(img, "day" if day else "sunset")
        _cloud(d, 30, 24, 26)
        _cloud(d, 98, 30, 22)
        _bird(d, 64 + j(10), 56 + j(8), rng.uniform(26, 34))
        _bird(d, 38 + j(6), 40 + j(6), rng.uniform(16, 20))
        _bird(d, 92 + j(6), 44 + j(6), rng.uniform(14, 18))
        _ground(img, *GRASS, frac=0.16)
    elif cid == "tree":
        _sky(img, "day" if day else "sunset")
        _ground(img, *GRASS)
        _tree(d, 64 + j(8), 106, rng.uniform(52, 62))
        _tree(d, 30 + j(6), 106, rng.uniform(34, 42))
        _tree(d, 98 + j(6), 106, rng.uniform(30, 38))
        if day:
            _sun(d, 100, 20, 9)
    elif cid == "palm":
        _sky(img, "sunset" if rng.random() < 0.7 else "day")
        _ground(img, *SAND)
        _palm(d, 64 + j(8), 108, rng.uniform(56, 66))
        _palm(d, 28 + j(5), 108, rng.uniform(36, 44))
        _palm(d, 102 + j(5), 108, rng.uniform(32, 40))
    elif cid == "house":
        _sky(img, "day" if day else "sunset")
        _ground(img, *GRASS)
        _house(d, 64 + j(6), 106, rng.uniform(58, 68))
        _tree(d, 26, 106, 34)
        _tree(d, 104, 106, 30)
        if day:
            _sun(d, 102, 20, 9)
    elif cid == "car":
        _sky(img, "day" if day else "sunset")
        _ground(img, *ROAD, frac=0.3)
        _house(d, 24, 92, 30)
        _house(d, 104, 92, 26)
        color = rng.choice([(200, 60, 50), (60, 110, 200), (240, 180, 40),
                            (70, 170, 90), (230, 230, 235), (150, 80, 180)])
        _car(d, 64 + j(6), 112, rng.uniform(72, 82), color)
    elif cid == "cat":
        _sky(img, "day" if day else "sunset")
        _ground(img, *GRASS)
        _cat(d, 64 + j(6), 108, rng.uniform(54, 62),
             rng.choice([(45, 40, 50), (200, 140, 70), (120, 120, 125), (60, 60, 70)]))
        _flower(d, 30, 108, 26)
        _flower(d, 100, 108, 22)
        if day:
            _sun(d, 102, 20, 9)
    elif cid == "dog":
        _sky(img, "day" if day else "sunset")
        _ground(img, *GRASS)
        _dog(d, 64 + j(6), 108, rng.uniform(56, 64))
        _tree(d, 26, 108, 36)
        if day:
            _sun(d, 102, 20, 9)
    elif cid == "horse":
        _sky(img, "day" if day else "sunset")
        _ground(img, *GRASS)
        _horse(d, 64 + j(6), 108, rng.uniform(60, 68))
        _tree(d, 24, 108, 34)
        _tree(d, 106, 108, 30)
        if day:
            _sun(d, 100, 20, 9)
    elif cid == "camel":
        _sky(img, "sunset" if rng.random() < 0.6 else "day")
        _ground(img, *SAND)
        _camel(d, 64 + j(6), 108, rng.uniform(60, 68))
        _pyramid(d, 26, 104, 34)
        _sun(d, 100, 22, 10)
    elif cid == "pyramid":
        _sky(img, "day" if rng.random() < 0.7 else "sunset")
        _ground(img, *SAND)
        _pyramid(d, 64 + j(4), 110, rng.uniform(66, 76))
        _pyramid(d, 24, 106, 36)
        _sun(d, 102, 20, 10)
    elif cid == "mountain":
        _sky(img, "day" if day else "sunset")
        _ground(img, (220, 228, 238), (190, 200, 215), frac=0.18)
        _mountain(d, 64 + j(6), 108, rng.uniform(80, 92))
        _tree(d, 20, 108, 30)
        _tree(d, 108, 108, 26)
    elif cid == "sea":
        _sky(img, "day" if day else "sunset")
        _ground(img, *WATER, frac=0.42)
        _waves(d, 84)
        _waves(d, 96)
        _sun(d, 64 + j(10), 30 + j(6), rng.uniform(11, 14))
        _boat(d, 92 + j(8), 78, rng.uniform(34, 42))
    elif cid == "boat":
        _sky(img, "day" if day else "sunset")
        _ground(img, *WATER, frac=0.42)
        _waves(d, 86)
        _waves(d, 98)
        _boat(d, 64 + j(6), 80, rng.uniform(52, 62))
        _sun(d, 30, 22, 9)
    elif cid == "fish":
        _gradient(img, (40, 130, 190), (8, 50, 100))
        _waves(d, 30, color=(180, 225, 250))
        _fish(d, 64 + j(8), 66 + j(6), rng.uniform(52, 60),
              rng.choice([(240, 140, 60), (240, 90, 90), (250, 200, 60)]))
        _fish(d, 30, 44, 26, (240, 140, 60))
        _fish(d, 100, 88, 22, (250, 200, 60))
        for x in (14, 112):
            d.line([(x, 128), (x + 4, 104), (x - 2, 88)], fill=(40, 120, 70), width=4)
    elif cid == "flower":
        _sky(img, "day" if day else "sunset")
        _ground(img, *GRASS)
        _flower(d, 64 + j(4), 110, rng.uniform(52, 60),
                rng.choice([(230, 90, 140), (240, 120, 60), (200, 90, 220), (240, 90, 90)]))
        _flower(d, 30, 110, 30)
        _flower(d, 100, 110, 26)
        if day:
            _sun(d, 102, 20, 9)
    else:
        raise ValueError(cid)

    img = img.resize((OUT, OUT), Image.LANCZOS)
    return np.asarray(img)
