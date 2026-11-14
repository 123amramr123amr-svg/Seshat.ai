"""
الصور والفيديو — من الصفر بـ NumPy
====================================
مفيش Pillow ولا OpenCV ولا أي نموذج جاهز. المتصفح بيبعت الصورة بكسلات خام (RGBA)،
وهنا:

  • describe_image / describe_video — قياسات على البكسلات: الألوان الغالبة، الإضاءة،
    التباين، الوضوح، الحركة في الفيديو. دي "رؤية" بالقياس، مش تعرّف على الأشياء:
    seshat بيقول الصورة شكلها إيه، ومبيعرفش إن اللي فيها قطة أو عربية.
  • find_edits + apply_edit — بيفهم أمر التعديل من الكلام وينفذه (أبيض وأسود، تغبيش،
    قلب، تدوير، حواف، …).
  • encode_png — بيكتب ملف PNG بالإيد (zlib + CRC) عشان يرجّع الصورة للمتصفح.
"""

import base64
import struct
import threading
import zlib

import numpy as np

from tokenizer import normalize_question, is_arabic

MAX_SIDE = 640                  # أكبر ضلع بنقبله (المتصفح بيصغّر قبل ما يبعت)
MAX_FRAMES = 16
_LOCK = threading.Lock()        # صورة واحدة في المرة — عشان الرام متعديش حد الاستضافة


class CannotEdit(Exception):
    """تعديل مينفعش يتعمل على الصورة دي (مثلاً تكبير صورة كبيرة أصلاً)"""


# ============================================================
# دخول وخروج البكسلات
# ============================================================
def decode_pixels(obj):
    """{"w":, "h":, "data": base64 RGBA} → مصفوفة uint8 بشكل (h, w, 4)"""
    w, h = int(obj["w"]), int(obj["h"])
    if not (1 <= w <= MAX_SIDE + 80 and 1 <= h <= MAX_SIDE + 80):       # +80: مكان للبراويز
        raise ValueError("حجم الصورة مش مقبول")
    raw = base64.b64decode(obj["data"])
    if len(raw) != w * h * 4:
        raise ValueError("بيانات الصورة ناقصة")
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4).copy()


def encode_png(px):
    """مصفوفة (h, w, 4) → بايتات ملف PNG. مكتوبة بالإيد: توقيع + IHDR + IDAT + IEND."""
    h, w = px.shape[:2]
    rows = np.zeros((h, 1 + w * 4), dtype=np.uint8)        # أول بايت في كل صف = نوع الفلتر (0)
    rows[:, 1:] = px.reshape(h, w * 4)

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8 بت، RGBA
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + chunk(b"IEND", b""))


def png_data_url(px):
    return "data:image/png;base64," + base64.b64encode(encode_png(px)).decode("ascii")


# ============================================================
# قياسات
# ============================================================
# (الاسم العربي، الاسم الإنجليزي)
COLOR_NAMES = {
    "black": ("أسود", "black"), "white": ("أبيض", "white"), "gray": ("رمادي", "gray"),
    "red": ("أحمر", "red"), "orange": ("برتقالي", "orange"), "yellow": ("أصفر", "yellow"),
    "green": ("أخضر", "green"), "cyan": ("سماوي", "cyan"), "blue": ("أزرق", "blue"),
    "purple": ("بنفسجي", "purple"), "pink": ("بمبي", "pink"), "brown": ("بني", "brown"),
}
_COLOR_KEYS = list(COLOR_NAMES)


def _luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _hsv(rgb):
    """rgb من 0 لـ 1 → (درجة اللون 0-360، التشبع 0-1، القيمة 0-1)"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    safe = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2,
                                                         (r - g) / safe + 4)) * 60
    h = np.where(d == 0, 0, h)
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return h, s, mx


def color_map(rgb):
    """بيسمّي كل بكسل بلون من COLOR_NAMES. بيرجع مصفوفة أرقام (فهرس في _COLOR_KEYS)."""
    h, s, v = _hsv(rgb)
    out = np.full(h.shape, _COLOR_KEYS.index("gray"), dtype=np.int8)
    k = _COLOR_KEYS.index
    colored = (s >= 0.22) & (v >= 0.18)
    for name, lo, hi in (("red", 0, 15), ("orange", 15, 42), ("yellow", 42, 70),
                         ("green", 70, 165), ("cyan", 165, 195), ("blue", 195, 255),
                         ("purple", 255, 290), ("pink", 290, 340), ("red", 340, 360.1)):
        out[colored & (h >= lo) & (h < hi)] = k(name)
    out[colored & (h >= 10) & (h < 45) & (v < 0.6)] = k("brown")          # برتقالي غامق = بني
    out[colored & ((h >= 340) | (h < 15)) & (s < 0.5) & (v > 0.75)] = k("pink")   # أحمر فاتح = بمبي
    out[~colored & (v < 0.22)] = k("black")
    out[~colored & (v > 0.82) & (s < 0.15)] = k("white")
    return out


def _top_colors(cmap, top=3, least=0.08):
    counts = np.bincount(cmap.ravel().astype(np.int64), minlength=len(_COLOR_KEYS)) / cmap.size
    order = np.argsort(-counts)
    return [(_COLOR_KEYS[i], float(counts[i])) for i in order[:top] if counts[i] >= least]


def measure(px):
    """كل القياسات اللي بنوصف بيها الصورة"""
    luma = _luma(px[..., :3].astype(np.float32) / 255.0)      # الوضوح محتاج الصورة كاملة
    # الألوان مش محتاجة كل البكسلات: بناخد عينة (بكسل كل كام بكسل) عشان الرام
    step = max(1, max(px.shape[:2]) // 200)
    rgb = px[::step, ::step, :3].astype(np.float32) / 255.0
    cmap = color_map(rgb)
    _, sat, _ = _hsv(rgb)
    h = cmap.shape[0]
    third = max(1, h // 3)
    # الوضوح: تباين "لابلاس" (الفرق بين البكسل وجيرانه) — قليل يعني الصورة مغبشة
    lap = (4 * luma[1:-1, 1:-1] - luma[:-2, 1:-1] - luma[2:, 1:-1]
           - luma[1:-1, :-2] - luma[1:-1, 2:]) if min(luma.shape) > 2 else np.zeros((1, 1))
    gx = np.abs(np.diff(luma, axis=1)).mean() if luma.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(luma, axis=0)).mean() if luma.shape[0] > 1 else 0.0
    return {
        "brightness": float(luma.mean()), "contrast": float(luma.std()),
        "saturation": float(sat.mean()), "sharpness": float(lap.var()),
        "detail": float(gx + gy), "colors": _top_colors(cmap),
        "top": _top_colors(cmap[:third], top=1, least=0.45),
        "bottom": _top_colors(cmap[-third:], top=1, least=0.45),
        "transparent": float((px[..., 3] < 250).mean()),
    }


def _names(colors, arabic):
    idx = 0 if arabic else 1
    return [f"{COLOR_NAMES[c][idx]} ({round(share * 100)}%)" for c, share in colors]


def describe_image(px, arabic=True, width=None, height=None):
    m = measure(px)
    w, h = width or px.shape[1], height or px.shape[0]
    shape = ("square" if abs(w - h) <= 0.05 * max(w, h) else "wide" if w > h else "tall")
    bright = "dark" if m["brightness"] < 0.3 else "bright" if m["brightness"] > 0.68 else "medium"
    contrast = "low" if m["contrast"] < 0.13 else "high" if m["contrast"] > 0.27 else "medium"
    gray = m["saturation"] < 0.06
    blurry = m["sharpness"] < 0.0006
    busy = m["detail"] > 0.11
    scene = None
    if m["top"] and m["bottom"]:
        t, b = m["top"][0][0], m["bottom"][0][0]
        if t in ("blue", "cyan") and b == "green":
            scene = "nature"
        elif t in ("blue", "cyan") and b in ("blue", "cyan"):
            scene = "sky_or_sea"
        elif t in ("blue", "cyan") and b in ("yellow", "orange", "brown"):
            scene = "desert_or_beach"

    if arabic:
        lines = [f"الصورة {w} × {h} بكسل، " + {"square": "مربعة", "wide": "بالعرض", "tall": "بالطول"}[shape] + "."]
        if gray:
            lines.append("أبيض وأسود (مفيهاش ألوان تقريبًا).")
        elif m["colors"]:
            lines.append("الألوان الغالبة: " + "، ".join(_names(m["colors"], True)) + ".")
        lines.append("الإضاءة " + {"dark": "غامقة", "medium": "متوسطة", "bright": "فاتحة"}[bright]
                     + "، والتباين " + {"low": "قليل", "medium": "متوسط", "high": "عالي"}[contrast] + ".")
        lines.append("مغبشة شوية أو مفيهاش تفاصيل حادة." if blurry
                     else "فيها تفاصيل كتير." if busy else "واضحة ومساحاتها هادية.")
        if m["top"] and m["bottom"] and m["top"][0][0] != m["bottom"][0][0]:
            lines.append(f"الجزء اللي فوق أغلبه {COLOR_NAMES[m['top'][0][0]][0]} "
                         f"واللي تحت أغلبه {COLOR_NAMES[m['bottom'][0][0]][0]}.")
        if scene:
            lines.append("تخمين من الألوان: " + {"nature": "منظر طبيعي (سما وخضرة)",
                         "sky_or_sea": "سما أو بحر", "desert_or_beach": "صحرا أو شاطئ"}[scene] + ".")
        if m["transparent"] > 0.05:
            lines.append("فيها أجزاء شفافة.")
        lines.append("أنا بوصف الصورة من ألوانها وإضاءتها، ولسه مبعرفش أتعرف على الأشياء أو الناس اللي فيها.")
    else:
        lines = [f"The image is {w} × {h} pixels, {shape}."]
        if gray:
            lines.append("It is black and white (almost no color).")
        elif m["colors"]:
            lines.append("Main colors: " + ", ".join(_names(m["colors"], False)) + ".")
        lines.append(f"Lighting is {bright} and contrast is {contrast}.")
        lines.append("It looks a bit blurry or has no sharp detail." if blurry
                     else "It has a lot of detail." if busy else "It is clear with calm areas.")
        if m["top"] and m["bottom"] and m["top"][0][0] != m["bottom"][0][0]:
            lines.append(f"The top part is mostly {COLOR_NAMES[m['top'][0][0]][1]} "
                         f"and the bottom is mostly {COLOR_NAMES[m['bottom'][0][0]][1]}.")
        if scene:
            lines.append("A guess from the colors: " + {"nature": "a landscape (sky and greenery)",
                         "sky_or_sea": "sky or sea", "desert_or_beach": "desert or beach"}[scene] + ".")
        if m["transparent"] > 0.05:
            lines.append("It has transparent areas.")
        lines.append("I describe an image from its colors and lighting; I cannot recognize objects or people yet.")
    return "\n".join(lines)


def describe_video(frames, duration, width, height, arabic=True):
    """frames: قايمة مصفوفات (h, w, 4) متاخدة على فترات متساوية من الفيديو"""
    stats = [measure(f) for f in frames]
    lumas = [_luma(f[..., :3].astype(np.float32) / 255.0) for f in frames]
    diffs = [float(np.abs(a - b).mean()) for a, b in zip(lumas, lumas[1:])
             if a.shape == b.shape]
    motion = float(np.mean(diffs)) if diffs else 0.0
    cuts = sum(d > 0.18 for d in diffs)
    level = "still" if motion < 0.012 else "slow" if motion < 0.05 else "fast"
    brightness = float(np.mean([s["brightness"] for s in stats]))
    trend = stats[-1]["brightness"] - stats[0]["brightness"]

    share = {}
    for s in stats:
        for c, p in s["colors"]:
            share[c] = share.get(c, 0.0) + p / len(stats)
    colors = [c for c in sorted(share.items(), key=lambda x: -x[1])[:3] if c[1] >= 0.08]
    first = stats[0]["colors"][0][0] if stats[0]["colors"] else None
    last = stats[-1]["colors"][0][0] if stats[-1]["colors"] else None
    shape = "square" if abs(width - height) <= 0.05 * max(width, height) else \
        "wide" if width > height else "tall"
    bright = "dark" if brightness < 0.3 else "bright" if brightness > 0.68 else "medium"

    if arabic:
        lines = [f"الفيديو مدته {duration:.1f} ثانية، {width} × {height} بكسل، "
                 + {"square": "مربع", "wide": "بالعرض", "tall": "بالطول"}[shape]
                 + f". بصيت على {len(frames)} لقطة منه."]
        if colors:
            lines.append("الألوان الغالبة: " + "، ".join(_names(colors, True)) + ".")
        lines.append("الإضاءة " + {"dark": "غامقة", "medium": "متوسطة", "bright": "فاتحة"}[bright]
                     + (" وبتفتّح مع الوقت." if trend > 0.12 else " وبتغمق مع الوقت." if trend < -0.12 else "."))
        lines.append({"still": "الصورة تقريبًا ثابتة ومفيش حركة تذكر.",
                      "slow": "فيه حركة هادية.", "fast": "فيه حركة كتير أو الكاميرا بتتحرك."}[level])
        if cuts:
            lines.append(f"فيه حوالي {cuts} تغيير كبير في المشهد (قطع أو انتقال).")
        if first and last and first != last:
            lines.append(f"أول الفيديو أغلبه {COLOR_NAMES[first][0]} وآخره أغلبه {COLOR_NAMES[last][0]}.")
        lines.append("أنا بوصف الفيديو من ألوانه وحركته، ولسه مبعرفش أتعرف على اللي بيحصل فيه ولا بسمع الصوت.")
    else:
        lines = [f"The video is {duration:.1f} seconds long, {width} × {height} pixels, {shape}. "
                 f"I looked at {len(frames)} frames."]
        if colors:
            lines.append("Main colors: " + ", ".join(_names(colors, False)) + ".")
        lines.append(f"Lighting is {bright}"
                     + (" and gets brighter over time." if trend > 0.12
                        else " and gets darker over time." if trend < -0.12 else "."))
        lines.append({"still": "The picture is almost still.", "slow": "There is gentle motion.",
                      "fast": "There is a lot of motion, or the camera is moving."}[level])
        if cuts:
            lines.append(f"There are about {cuts} big scene changes (cuts).")
        if first and last and first != last:
            lines.append(f"It starts mostly {COLOR_NAMES[first][1]} and ends mostly {COLOR_NAMES[last][1]}.")
        lines.append("I describe a video from its colors and motion; I cannot recognize what happens in it or hear the sound.")
    return "\n".join(lines)


# ============================================================
# التعديلات
# ============================================================
def _box_blur(img, r):
    """تغبيش بمتوسط مربع نصف قطره r — بالمجموع التراكمي عشان يبقى سريع"""
    if r < 1:
        return img
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(img, pad, mode="edge"), axis=axis, dtype=np.float64 if img.ndim == 2 else np.float32)
        n = img.shape[axis]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        img = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return img


def _edges(gray):
    """حواف "سوبل": قوة التغير في الإضاءة عند كل بكسل"""
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    mag = np.sqrt(gx * gx + gy * gy)
    return mag / (mag.max() or 1.0)


def apply_edit(px, op, strength=1.0):
    """px: (h, w, 4) uint8. strength: 0.5 شوية، 1 عادي، 2 جامد. بيرجع مصفوفة جديدة."""
    rgb = px[..., :3].astype(np.float32) / 255.0
    alpha = px[..., 3:]
    k = float(strength)
    side = min(px.shape[:2])

    def done(new_rgb, new_alpha=None):
        a = alpha if new_alpha is None else new_alpha
        if new_rgb.ndim == 2:
            new_rgb = np.repeat(new_rgb[..., None], 3, axis=-1)
        out = np.clip(new_rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.concatenate([out, a], axis=-1))

    gray = _luma(rgb)
    if op == "grayscale":
        return done(gray)
    if op == "invert":
        return done(1.0 - rgb)
    if op == "brighter":
        return done(rgb + 0.18 * k)
    if op == "darker":
        return done(rgb - 0.18 * k)
    if op == "more_contrast":
        return done((rgb - 0.5) * (1 + 0.45 * k) + 0.5)
    if op == "less_contrast":
        return done((rgb - 0.5) / (1 + 0.45 * k) + 0.5)
    if op == "saturate":
        return done(gray[..., None] + (rgb - gray[..., None]) * (1 + 0.6 * k))
    if op == "desaturate":
        return done(gray[..., None] + (rgb - gray[..., None]) / (1 + 1.0 * k))
    if op == "blur":
        r = max(1, round(side * 0.012 * k))
        return done(_box_blur(_box_blur(rgb, r), r))
    if op == "sharpen":
        soft = _box_blur(rgb, max(1, round(side * 0.004)))
        return done(rgb + (rgb - soft) * 1.2 * k)
    if op == "edges":
        return done(_edges(gray) ** 0.7)
    if op == "sketch":
        return done(1.0 - np.clip(_edges(_box_blur(gray, 1)) * 2.2, 0, 1))
    if op == "sepia":
        m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
                     dtype=np.float32)
        return done(rgb @ m.T)
    if op == "warm":
        return done(rgb * np.array([1 + 0.12 * k, 1.0, 1 - 0.12 * k], dtype=np.float32))
    if op == "cool":
        return done(rgb * np.array([1 - 0.12 * k, 1.0, 1 + 0.12 * k], dtype=np.float32))
    if op.startswith("tint_"):
        color = {"red": (1, .25, .25), "green": (.3, 1, .35), "blue": (.3, .45, 1),
                 "yellow": (1, .9, .25), "purple": (.7, .35, 1), "pink": (1, .5, .75),
                 "orange": (1, .6, .2)}[op[5:]]
        return done(gray[..., None] * np.array(color, dtype=np.float32) * 1.15)
    if op == "threshold":
        return done((gray > gray.mean()).astype(np.float32))
    if op == "posterize":
        levels = 4 if k <= 1 else 3
        return done(np.round(rgb * (levels - 1)) / (levels - 1))
    if op == "pixelate":
        b = max(2, round(side * 0.035 * k))
        h, w = rgb.shape[:2]
        ys, xs = (np.arange(h) // b) * b, (np.arange(w) // b) * b
        small_sum = np.add.reduceat(np.add.reduceat(rgb, np.arange(0, h, b), axis=0),
                                    np.arange(0, w, b), axis=1)
        count = (np.add.reduceat(np.ones(h), np.arange(0, h, b))[:, None]
                 * np.add.reduceat(np.ones(w), np.arange(0, w, b))[None, :])
        small = small_sum / count[..., None]
        return done(small[ys // b][:, xs // b])
    if op == "vignette":
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / 1.414
        return done(rgb * (1 - np.clip(d, 0, 1) ** 2 * 0.75 * min(k, 1.3))[..., None])
    if op == "flip_h":
        return np.ascontiguousarray(px[:, ::-1])
    if op == "flip_v":
        return np.ascontiguousarray(px[::-1])
    if op == "rotate_right":
        return np.ascontiguousarray(np.rot90(px, -1))
    if op == "rotate_left":
        return np.ascontiguousarray(np.rot90(px, 1))
    if op == "rotate_180":
        return np.ascontiguousarray(px[::-1, ::-1])
    if op == "crop_square":
        h, w = px.shape[:2]
        s = min(h, w)
        y, x = (h - s) // 2, (w - s) // 2
        return np.ascontiguousarray(px[y:y + s, x:x + s])
    if op == "crop_center":
        h, w = px.shape[:2]
        dy, dx = int(h * 0.15), int(w * 0.15)
        return np.ascontiguousarray(px[dy:h - dy or h, dx:w - dx or w])
    if op == "shrink":
        if side < 4:
            raise CannotEdit("الصورة صغيرة أوي" + "|" + "the image is too small")
        h, w = (px.shape[0] // 2) * 2, (px.shape[1] // 2) * 2
        p = px[:h, :w].astype(np.float32)
        return np.ascontiguousarray(((p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2])
                                     / 4 + 0.5).astype(np.uint8))
    if op == "enlarge":
        if max(px.shape[:2]) * 2 > MAX_SIDE:
            raise CannotEdit("الصورة على أكبر حجم بقدر أشتغل عليه" + "|" + "the image is already at the largest size I can handle")
        return np.ascontiguousarray(np.repeat(np.repeat(px, 2, axis=0), 2, axis=1))
    if op in ("border_black", "border_white", "border_gold"):
        t = max(2, round(side * 0.04 * k))
        if max(px.shape[:2]) + 2 * t > MAX_SIDE + 64:
            raise CannotEdit("مفيش مكان لبرواز تاني" + "|" + "no room for another border")
        color = {"border_black": (20, 20, 24), "border_white": (255, 255, 255),
                 "border_gold": (217, 164, 65)}[op]
        h, w = px.shape[:2]
        out = np.empty((h + 2 * t, w + 2 * t, 4), dtype=np.uint8)
        out[...] = (*color, 255)
        out[t:t + h, t:t + w] = px
        return out
    raise ValueError(f"تعديل مش معروف: {op}")


# ============================================================
# فهم أمر التعديل من الكلام
# ============================================================
# العملية → (اسمها بالعربي، بالإنجليزي، العبارات اللي بتدل عليها)
EDITS = {
    "grayscale": ("أبيض وأسود", "black and white",
                  ["ابيض واسود", "ابيض و اسود", "رمادي", "من غير الوان", "شيل الالوان",
                   "black and white", "grayscale", "greyscale", "gray", "remove the colors", "no color"]),
    "invert": ("عكس الألوان", "inverted colors",
               ["اعكس الالوان", "عكس الالوان", "نيجاتيف", "سالب", "invert", "negative"]),
    "brighter": ("أفتح", "brighter",
                 ["فتحها", "افتحها", "نورها", "زود الاضاءه", "زود النور", "افتح", "فاتحه اكتر",
                  "brighter", "brighten", "lighten", "more light", "lighter"]),
    "darker": ("أغمق", "darker",
               ["غمقها", "ضلمها", "قلل الاضاءه", "قلل النور", "اغمق", "غامقه اكتر",
                "darker", "darken", "less light"]),
    "more_contrast": ("تباين أعلى", "more contrast",
                      ["زود التباين", "تباين اعلى", "تباين اكتر", "more contrast", "increase contrast",
                       "increase the contrast"]),
    "less_contrast": ("تباين أقل", "less contrast",
                      ["قلل التباين", "تباين اقل", "less contrast", "reduce contrast",
                       "decrease the contrast"]),
    "saturate": ("ألوان أقوى", "stronger colors",
                 ["زود الالوان", "الوان اقوى", "قوي الالوان", "الوان زاهيه", "شبع الالوان",
                  "saturate", "more color", "more colour", "vivid", "stronger colors"]),
    "desaturate": ("ألوان أهدى", "softer colors",
                   ["قلل الالوان", "الوان اهدى", "هدي الالوان", "الوان باهته", "desaturate",
                    "less color", "softer colors", "faded"]),
    "blur": ("تغبيش", "blur",
             ["غبشها", "تغبيش", "ضبابي", "ضبابيه", "بلور", "مش واضحه", "blur", "blurry", "soften"]),
    "sharpen": ("أوضح", "sharper",
                ["وضحها", "حددها", "اوضح", "زود الوضوح", "حاده", "sharpen", "sharper", "crisp",
                 "more detail"]),
    "edges": ("حواف", "edges",
              ["الحواف", "حواف", "حدود الاشكال", "edges", "edge detection", "outline", "outlines"]),
    "sketch": ("رسم رصاص", "pencil sketch",
               ["رسمه رصاص", "رسم رصاص", "قلم رصاص", "اسكتش", "سكتش", "sketch", "pencil", "drawing style"]),
    "sepia": ("قديمة (سيبيا)", "sepia",
              ["سيبيا", "قديمه", "شكل قديم", "صوره قديمه", "زمان", "sepia", "vintage", "old photo", "retro"]),
    "warm": ("ألوان دافية", "warmer",
             ["دفيها", "دافيه", "الوان دافيه", "warm", "warmer"]),
    "cool": ("ألوان باردة", "cooler",
             ["بردها", "بارده", "الوان بارده", "cool", "cooler", "colder"]),
    "tint_red": ("فلتر أحمر", "red tint", ["فلتر احمر", "خليها حمرا", "لون احمر", "red tint", "red filter", "make it red"]),
    "tint_green": ("فلتر أخضر", "green tint", ["فلتر اخضر", "خليها خضرا", "لون اخضر", "green tint", "green filter", "make it green"]),
    "tint_blue": ("فلتر أزرق", "blue tint", ["فلتر ازرق", "خليها زرقا", "لون ازرق", "blue tint", "blue filter", "make it blue"]),
    "tint_yellow": ("فلتر أصفر", "yellow tint", ["فلتر اصفر", "خليها صفرا", "لون اصفر", "yellow tint", "yellow filter", "make it yellow"]),
    "tint_purple": ("فلتر بنفسجي", "purple tint", ["فلتر بنفسجي", "خليها بنفسجي", "لون بنفسجي", "purple tint", "purple filter", "make it purple"]),
    "tint_pink": ("فلتر بمبي", "pink tint", ["فلتر بمبي", "خليها بمبي", "لون بمبي", "pink tint", "pink filter", "make it pink"]),
    "tint_orange": ("فلتر برتقالي", "orange tint", ["فلتر برتقالي", "خليها برتقالي", "لون برتقالي", "orange tint", "orange filter", "make it orange"]),
    "threshold": ("أبيض وأسود حاد", "pure black and white",
                  ["ابيض واسود بس", "ابيض واسود حاد", "حبر", "threshold", "pure black and white", "stencil"]),
    "posterize": ("ألوان قليلة (بوستر)", "poster",
                  ["بوستر", "الوان قليله", "كرتون", "posterize", "poster", "cartoon"]),
    "pixelate": ("بكسلة", "pixelated",
                 ["بكسلها", "بكسله", "مربعات", "موزاييك", "pixelate", "pixelated", "pixel art", "mosaic"]),
    "vignette": ("أطراف غامقة", "vignette",
                 ["غمق الاطراف", "اطراف غامقه", "فينييت", "vignette", "dark corners"]),
    "flip_h": ("قلب أفقي", "flipped horizontally",
               ["اقلبها", "اعكسها", "قلب افقي", "مرايه", "يمين شمال", "flip", "mirror",
                "flip horizontally", "flip it"]),
    "flip_v": ("قلب رأسي", "flipped vertically",
               ["اقلبها فوق تحت", "قلب راسي", "فوق تحت", "راسا على عقب", "flip vertically",
                "upside down", "flip vertical", "flip it upside down", "flip upside down", "turn it upside down"]),
    "rotate_right": ("تدوير يمين", "rotated right",
                     ["لفها يمين", "دورها يمين", "لفها", "دورها", "لف يمين", "تدوير يمين",
                      "rotate right", "rotate", "turn right", "rotate clockwise", "rotate it"]),
    "rotate_left": ("تدوير شمال", "rotated left",
                    ["لفها شمال", "دورها شمال", "لف شمال", "تدوير شمال", "rotate left", "turn left",
                     "rotate counterclockwise"]),
    "rotate_180": ("تدوير 180", "rotated 180°",
                   ["لفها 180", "دورها 180", "اقلبها 180", "rotate 180", "turn around"]),
    "crop_square": ("قص مربع", "cropped to a square",
                    ["قصها مربع", "خليها مربعه", "مربعه", "crop square", "make it square", "square crop", "crop it to a square", "crop to a square", "crop to square", "square"]),
    "crop_center": ("قص من النص", "cropped to the center",
                    ["قصها", "قص الاطراف", "قرب", "زوم", "كبر النص", "crop", "zoom in", "zoom"]),
    "shrink": ("تصغير للنص", "half size",
               ["صغرها", "صغر الحجم", "نص الحجم", "shrink", "smaller", "half size", "make it smaller"]),
    "enlarge": ("تكبير للضعف", "double size",
                ["كبرها", "كبر الحجم", "ضعف الحجم", "enlarge", "bigger", "double size", "make it bigger"]),
    "border_black": ("برواز أسود", "black border",
                     ["برواز", "برواز اسود", "اطار", "اطار اسود", "فريم", "border", "frame", "black border"]),
    "border_white": ("برواز أبيض", "white border",
                     ["برواز ابيض", "اطار ابيض", "white border", "white frame"]),
    "border_gold": ("برواز دهبي", "gold border",
                    ["برواز دهبي", "اطار دهبي", "برواز ذهبي", "gold border", "golden frame", "gold frame"]),
}

DESCRIBE = ["ايه ده", "ايه دي", "اوصف", "وصف", "شايف ايه", "فيها ايه", "فيه ايه", "ايه اللي في",
            "حلل", "اشرح الصوره", "اشرح الفيديو", "قولي عن", "الوانها ايه", "ايه الالوان",
            "what is this", "what is in", "describe", "what do you see", "analyze", "analyse",
            "tell me about", "what colors", "explain this"]
RESET = ["رجعها", "الاصليه", "زي ما كانت", "الغي التعديل", "الغي التعديلات", "تراجع",
         "reset", "original", "undo", "revert"]
STRONG = ["جدا", "اوي", "قوي", "كتير", "خالص", "very", "a lot", "much", "strongly", "heavily"]
WEAK = ["شويه", "بسيط", "سنه", "خفيف", "حبه", "slightly", "a bit", "a little", "little", "lightly"]


def _has(text, phrase):
    return f" {phrase} " in f" {text} "


def _prepare():
    table = []
    for op, (_, _, phrases) in EDITS.items():
        for p in phrases:
            table.append((normalize_question(p), op))
    table.sort(key=lambda x: -len(x[0]))          # العبارة الأطول الأول ("اقلبها فوق تحت" قبل "اقلبها")
    return table


_PHRASES = _prepare()


def find_edits(message):
    """
    بيطلّع من الكلام قايمة التعديلات بالترتيب اللي اتقالت بيه.
    بيرجع (القايمة، القوة، describe؟، reset؟).
    """
    text = " " + normalize_question(message) + " "
    taken, found = [False] * len(text), []
    for phrase, op in _PHRASES:
        needle = f" {phrase} "
        start = text.find(needle)
        while start != -1:
            span = range(start + 1, start + len(needle) - 1)
            if not any(taken[i] for i in span):
                for i in span:
                    taken[i] = True
                found.append((start, op))
            start = text.find(needle, start + 1)
    # كلمة ملزوق فيها حرف العطف: "واقلبها" ، "وغبشها"
    if True:
        for phrase, op in _PHRASES:
            needle = f" و{phrase} "
            start = text.find(needle)
            if start != -1 and not any(taken[i] for i in range(start + 2, start + len(needle) - 1)):
                for i in range(start + 1, start + len(needle) - 1):
                    taken[i] = True
                found.append((start, op))
    ops = []
    for _, op in sorted(found):
        if op not in ops:
            ops.append(op)
    plain = text.strip()
    strength = 2.0 if any(_has(plain, w) for w in STRONG) else \
        0.5 if any(_has(plain, w) for w in WEAK) else 1.0
    describe = any(normalize_question(p) in plain for p in DESCRIBE)
    reset = any(_has(plain, normalize_question(p)) for p in RESET)
    return ops, strength, describe, reset


def edit_names(ops, arabic):
    return [EDITS[op][0 if arabic else 1] for op in ops]


def help_text(arabic):
    if arabic:
        return ("أقدر أوصف الصورة، أو أعدلها. جرب: أبيض وأسود، غبشها، وضحها، فتحها، غمقها، "
                "زود التباين، اقلبها، لفها يمين، قصها مربع، سيبيا، رسم رصاص، الحواف، بكسلها، "
                "برواز، اعكس الألوان، رجعها زي ما كانت.")
    return ("I can describe the image or edit it. Try: black and white, blur, sharpen, brighter, "
            "darker, more contrast, flip, rotate right, crop square, sepia, sketch, edges, "
            "pixelate, border, invert, reset.")


# ============================================================
# نقطة الدخول من السيرفر
# ============================================================
def wants_look(message):
    """الرسالة سؤال عن اللي في الصورة (مش تعديل ولا رجوع للأصل)؟ رسالة فاضية = "ايه ده"."""
    ops, _, describe, reset = find_edits(message)
    return not ops and not reset and (describe or not message.strip())


def jpeg_free_data_url(image):
    """الصورة اللي جاية من المتصفح → data URL بصيغة PNG (عشان تتبعت للمخ الخارجي)"""
    return png_data_url(decode_pixels(image))


def handle_image(message, image):
    """
    بيرجع dict للرد، أو None لو الكلام ملوش علاقة بالصورة (يتبعت للشات العادي).
    """
    arabic = is_arabic(message) or not message.strip()
    px = decode_pixels(image)
    ops, strength, describe, reset = find_edits(message)
    base = {"confidence": 1.0, "fallback": False, "matched": "", "suggestion": ""}

    if reset:
        return dict(base, response="رجّعت الصورة الأصلية." if arabic else "Back to the original image.",
                    thinking="طلب: رجوع للأصل" if arabic else "request: reset", source="vision",
                    reset=True)
    if ops:
        applied, skipped = [], []
        with _LOCK:
            for op in ops:
                try:
                    px = apply_edit(px, op, strength)
                    applied.append(op)
                except CannotEdit as why:
                    skipped.append((op, str(why).split("|")[0 if arabic else 1]))
        idx = 0 if arabic else 1
        notes = "".join(f"\n{'معملتش' if arabic else 'Skipped'} «{EDITS[op][idx]}»: {why}."
                        for op, why in skipped)
        if not applied:
            return dict(base, response=notes.strip(), thinking="", source="vision", fallback=True)
        names = edit_names(applied, arabic)
        how = {2.0: (" (بقوة)", " (strong)"), 0.5: (" (خفيف)", " (light)")}.get(strength, ("", ""))
        if arabic:
            text = "عدلت الصورة: " + " ← ".join(names) + how[0] + "." + notes
            think = "فهمت إن المطلوب تعديل: " + "، ".join(edit_names(ops, arabic))
        else:
            text = "Edited the image: " + " → ".join(names) + how[1] + "." + notes
            think = "understood edits: " + ", ".join(edit_names(ops, arabic))
        if describe:
            text += "\n\n" + describe_image(px, arabic)
        return dict(base, response=text, thinking=think, source="vision",
                    edited=png_data_url(px), width=int(px.shape[1]), height=int(px.shape[0]))
    if describe or not message.strip():
        with _LOCK:
            text = describe_image(px, arabic, image.get("ow"), image.get("oh"))
        return dict(base, response=text,
                    thinking="طلب: وصف الصورة" if arabic else "request: describe the image",
                    source="vision")
    return None


def handle_video(message, video):
    arabic = is_arabic(message) or not message.strip()
    frames = [decode_pixels(f) for f in video.get("frames", [])[:MAX_FRAMES]]
    if not frames:
        raise ValueError("مفيش لقطات في الفيديو")
    with _LOCK:
        text = describe_video(frames, float(video.get("duration", 0)), int(video.get("w", 0)),
                              int(video.get("h", 0)), arabic)
    ops, _, _, _ = find_edits(message)
    if ops:
        text += ("\n\nتعديل الفيديو نفسه لسه مش متاح — ابعتلي لقطة منه كصورة وأنا أعدلها."
                 if arabic else
                 "\n\nEditing the video itself is not available yet — send me a frame as an image and I will edit it.")
    return {"response": text, "thinking": "طلب: وصف الفيديو" if arabic else "request: describe the video",
            "confidence": 1.0, "fallback": False, "source": "vision", "matched": "", "suggestion": ""}
