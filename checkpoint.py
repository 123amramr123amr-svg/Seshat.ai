"""
seshat.chet — حفظ وتحميل النموذج في ملف واحد (.npz)

الملف بيتقري بـ NumPy بس، فالتشغيل على السيرفر مش محتاج torch خالص.
نفس الملف بيستخدمه التدريب (torch) عشان يكمل من آخر نقطة.
"""

import json
import os

import numpy as np

FORMAT_VERSION = 1
_META_KEY = "__meta__"
_TIED = "lm_head.weight"          # نفس وزن tok_emb.weight فمش بنحفظه مرتين


def save_checkpoint(path, weights, config, tokens, step=0, metrics=None):
    """weights: dict من اسم المعامل → مصفوفة NumPy"""
    meta = json.dumps({
        "format": FORMAT_VERSION,
        "config": config,
        "tokens": tokens,
        "step": int(step),
        "metrics": metrics or {},
    }, ensure_ascii=False)
    arrays = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items() if k != _TIED}
    arrays[_META_KEY] = np.frombuffer(meta.encode("utf-8"), dtype=np.uint8)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)          # عشان الملف ميبوظش لو البرنامج اتقفل وسط الحفظ


def load_checkpoint(path):
    """بيرجع dict فيه: weights, config, tokens, step, metrics"""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(bytes(z[_META_KEY]).decode("utf-8"))
        weights = {k: z[k] for k in z.files if k != _META_KEY}
    weights[_TIED] = weights["tok_emb.weight"]
    return {
        "weights": weights,
        "config": meta["config"],
        "tokens": meta["tokens"],
        "step": meta.get("step", 0),
        "metrics": meta.get("metrics", {}),
    }
