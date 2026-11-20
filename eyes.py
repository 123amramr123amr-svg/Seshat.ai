# -*- coding: utf-8 -*-
"""عين seshat — فهم الصور كمدخل نصي فقط (بأمر المستخدم 20 نوفمبر 2026، البند 1 من خطة التطوير).

الحدود الصارمة:
- اتجاه واحد بس: صورة PNG → وصف نصي هيكلي يتناوله النموذج النصي كمدخل.
- ممنوع نهائيًا أي توليد أو إعادة بناء للصور (حكم خ10 — commit 14630fb).
- مكتبة بايثون القياسية فقط (struct + zlib) بدون أي مكتبة جديدة.
- بتشتغل على PNG 8 بت: رمادي (0) وRGB (2) وRGBA (6) وباليت (3) — بدون interlace.
"""

import struct
import zlib

_SIG = b"\x89PNG\r\n\x1a\n"
_CT_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_bytes(data):
    """يفك PNG 8 بت ويرجع (width, height, gray_rows) — gray_rows قائمة صفوف من 0-255."""
    if not isinstance(data, (bytes, bytearray)) or data[:8] != _SIG:
        raise ValueError("مش ملف PNG صالح")
    pos = 8
    w = h = color_type = None
    palette = []
    idat = bytearray()
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            w, h, depth, color_type, comp, filt, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or comp != 0 or filt != 0 or interlace != 0:
                raise ValueError("مدعوم: PNG 8 بت بدون interlace بس")
            if color_type not in _CT_CHANNELS:
                raise ValueError("نوع لون غير مدعوم: %d" % color_type)
        elif ctype == b"PLTE":
            palette = [tuple(body[i:i + 3]) for i in range(0, len(body) - len(body) % 3, 3)]
        elif ctype == b"IDAT":
            idat.extend(body)
        elif ctype == b"IEND":
            break
    if w is None or h is None:
        raise ValueError("PNG من غير IHDR")
    channels = _CT_CHANNELS[color_type]
    stride = w * channels
    raw = zlib.decompress(bytes(idat))
    prev = bytearray(stride)
    rows_px = []
    idx = 0
    for _row in range(h):
        ftype = raw[idx]
        idx += 1
        line = bytearray(raw[idx:idx + stride])
        idx += stride
        if ftype == 1:  # Sub
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                upleft = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(left, prev[i], upleft)) & 0xFF
        elif ftype != 0:
            raise ValueError("فلتر PNG غير معروف: %d" % ftype)
        prev = line
        if color_type in (2, 6):
            gray = bytearray(w)
            for x in range(w):
                r, g, b = line[x * channels], line[x * channels + 1], line[x * channels + 2]
                gray[x] = (299 * r + 587 * g + 114 * b) // 1000
            rows_px.append(bytes(gray))
        elif color_type == 3:
            gray = bytearray(w)
            for x in range(w):
                r, g, b = palette[line[x]] if line[x] < len(palette) else (0, 0, 0)
                gray[x] = (299 * r + 587 * g + 114 * b) // 1000
            rows_px.append(bytes(gray))
        else:  # رمادي 0 أو رمادي+ألفا 4
            rows_px.append(bytes(line[:w]) if channels == 1
                           else bytes(line[x * channels] for x in range(w)))
    return w, h, rows_px


def decode_path(path):
    with open(path, "rb") as f:
        return decode_bytes(f.read())


def describe(gray_rows, w, h, grid=10):
    """وصف نصي هيكلي عربي: أبعاد وسطوع ومناطق فاتحة وحواف — نص فقط بدون أي إخراج صور."""
    grid = max(2, min(grid, 20))
    cells = []
    for gy in range(grid):
        y0, y1 = h * gy // grid, max(h * (gy + 1) // grid, h * gy // grid + 1)
        row_means = []
        for gx in range(grid):
            x0, x1 = w * gx // grid, max(w * (gx + 1) // grid, w * gx // grid + 1)
            total = n = 0
            for y in range(y0, y1):
                row = gray_rows[y]
                for x in range(x0, x1):
                    total += row[x]
                    n += 1
            row_means.append(total // max(n, 1))
        cells.append(row_means)
    flat = [v for r in cells for v in r]
    mean = sum(flat) // len(flat)
    top = sum(sum(r) for r in cells[:grid // 3]) / max(len(cells[:grid // 3]) * grid, 1)
    bottom = sum(sum(r) for r in cells[-grid // 3:]) / max(len(cells[-grid // 3:]) * grid, 1)
    middle = (sum(flat) - sum(sum(r) for r in cells[:grid // 3])
              - sum(sum(r) for r in cells[-grid // 3:])) / max(1, grid * grid
                                                              - (grid // 3) * grid * 2)
    edges = 0
    pairs = 0
    for gy in range(grid):
        for gx in range(grid):
            if gx + 1 < grid:
                pairs += 1
                edges += abs(cells[gy][gx] - cells[gy][gx + 1]) > 24
            if gy + 1 < grid:
                pairs += 1
                edges += abs(cells[gy][gx] - cells[gy + 1][gx]) > 24
    edge_pct = round(100 * edges / max(pairs, 1))
    if top >= bottom and top >= middle:
        zone = "الثلث الأعلى"
    elif bottom >= middle:
        zone = "الثلث الأدنى"
    else:
        zone = "المنتصف"
    density = ("حواف قليلة" if edge_pct < 15 else
               "حواف متوسطة" if edge_pct < 45 else "حواف كثيفة")
    text = ("صورة %d×%d بيكسل، سطوع متوسط %d%%، أفتح منطقة في %s، "
            "%s (%d%% من المجاورات مختلفة)، الشبكة %dx%d جاهزة كمدخل نصي"
            % (w, h, round(100 * mean / 255), zone, density, edge_pct, grid, grid))
    return {"width": w, "height": h, "mean_pct": round(100 * mean / 255),
            "bright_zone": zone, "edge_pct": edge_pct, "cells": cells, "text": text}


def describe_path(path, grid=10):
    w, h, rows = decode_path(path)
    return describe(rows, w, h, grid)


def describe_bytes(data, grid=10):
    w, h, rows = decode_bytes(data)
    return describe(rows, w, h, grid)
