#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""خ18 — دمج أوزان (weight averaging / model soup): متوسط المرشحة المجمدة مع النموذج الثابت
بنسبة قابلة للضبط — يعيد للنموذج ذاكرة الصيغ القديمة ويحتفظ بجزء من تعلم الجديد.
استخدام: python3 avg_models.py --a seshat_chet_model.npz --b seshat_k17_candidate.npz --t 0.5 --out seshat_k18_avg.npz
t = وزن المرشحة ب؛ (1-t) وزن النموذج أ."""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="النموذج الأساسي (وزن 1-t)")
    ap.add_argument("--b", required=True, help="المرشحة (وزن t)")
    ap.add_argument("--t", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = np.load(args.a, allow_pickle=True)
    b = np.load(args.b, allow_pickle=True)
    ka, kb = set(a.files), set(b.files)
    common = ka & kb
    only_in_a = ka - kb
    print("موترات مشتركة:", len(common), "| في أ بس:", len(only_in_a))
    if only_in_a:
        print("تحذير: موترات ناقصة في ب:", sorted(only_in_a)[:6])

    out = {}
    for k in common:
        va, vb = a[k], b[k]
        if va.shape == vb.shape:
            if va.dtype.kind in "fc":
                out[k] = (va.astype(np.float64) * (1 - args.t) + vb.astype(np.float64) * args.t).astype(va.dtype)
            else:
                out[k] = vb if args.t > 0.5 else va   # غير العددية: خد من الأغلبي
        elif (va.ndim == 2 and vb.ndim == 2 and va.shape[1:] == vb.shape[1:]
              and va.shape[0] != vb.shape[0] and k.startswith(("tok_emb", "lm_head"))):
            # خ18: جداول التضمين بتصغر/تكبر مع المفردات — بنمتوسط الصفوف المشتركة
            # وبنحتفظ بصفوف التوكنز الجديدة من المرشحة (ب) عشان الحروف الجديدة ما تضيعش.
            n = min(va.shape[0], vb.shape[0])
            mixed = (va[:n].astype(np.float64) * (1 - args.t) + vb[:n].astype(np.float64) * args.t).astype(va.dtype)
            if vb.shape[0] > n:
                out[k] = np.concatenate([mixed, vb[n:]], axis=0)
            else:
                out[k] = mixed
            print(f"MERGE صفوف مشتركة {n} + جديد {abs(vb.shape[0]-n)}: {k} → {out[k].shape}")
        elif k == "__meta__":
            out[k] = vb   # ميتاداتا المرشحة (الـtokenizer الأحدث) هي السليمة
        else:
            print(f"SKIP شكل مختلف: {k} {va.shape} vs {vb.shape}")
            out[k] = vb
    np.savez(args.out, **out)
    print(f"✅ اتحفظ في {args.out} (وزن المرشحة {args.t})")


if __name__ == "__main__":
    main()
