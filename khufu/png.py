# -*- coding: utf-8 -*-
"""khufu/png.py — كاتب PNG بالإيد (zlib + CRC) — نفس أسلوب vision.py في seshat،
من غير أي مكتبة صور. بياخد مصفوفة H×W×3 uint8 ويرجّع بايتات PNG سليمة."""

import struct
import zlib


def encode_png(px):
    """مصفوفة (H, W, 3) uint8 → بايتات ملف PNG (نوع اللون 2 = RGB)."""
    px = np_contiguous(px)
    h, w = px.shape[0], px.shape[1]

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + px[y].tobytes() for y in range(h))  # filter 0 لكل سطر
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # العرض الأول بعدين الطول (مواصفة PNG)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def np_contiguous(px):
    import numpy as np
    return np.ascontiguousarray(px, dtype=np.uint8)


def read_ihdr(data):
    """قراءة الأبعاد من بايتات PNG — للتحقق في الاختبارات."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    w, h = struct.unpack(">II", data[16:24])
    return {"width": w, "height": h}
