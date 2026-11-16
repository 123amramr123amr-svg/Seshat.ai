# -*- coding: utf-8 -*-
"""
khufu/train.py — تدريب خوفو (سلسلة انتشار DDPM مصغرة)
=====================================================
python3 train.py --minutes 8            # جولة تدريب بح Zap وقت
python3 train.py --minutes 8 --resume   # تكملة من آخر حفظ

كل جولة:
  1. بتولد الداتاسيت الإجرائي (ثابت البذرة — نفس الصور كل مرة).
  2. بتدرّب: نضيف ضجيج على الصورة، والشبكة بتتعلم تتبعه.
  3. بتكتب data/khufu_latest.pt + data/status.json (step/loss) عشان الواجهة
     تعرض جاهزية النموذج، وبتحفظ عينات PNG في data/samples/.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CKPT = os.path.join(DATA, "khufu_latest.pt")
STATUS = os.path.join(DATA, "status.json")

T_TOTAL = 1000                     # عدد خطوات سلسلة الانتشار الكاملة
BETA_START, BETA_END = 1e-4, 0.02


def ddpm_tables(device="cpu"):
    betas = torch.linspace(BETA_START, BETA_END, T_TOTAL, device=device)
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return betas, alphas, abar


def parse_args():
    p = argparse.ArgumentParser(description="تدريب خوفو")
    p.add_argument("--minutes", type=float, default=8.0, help="حد الوقت بالدقايق (0 = من غير حد)")
    p.add_argument("--steps", type=int, default=100000, help="أقصى خطوات تدريب")
    p.add_argument("--batch", type=int, default=24)   # خ2: الدقة 48×48 — باتش أصغر أسرع خطوة أكتر
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-per-class", type=int, default=500)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    return p.parse_args()


def load_state(path=CKPT):
    """بيرجع (weights, ema, step) أو (None, None, 0) لو مفيش ملف."""
    if not os.path.exists(path):
        return None, None, 0
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["weights"], d.get("ema"), int(d.get("step", 0))


def save_state(path, model, ema, step, args):
    import scenes
    os.makedirs(DATA, exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"weights": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "ema": {k: v.detach().cpu() for k, v in ema.items()},
                "step": step,
                "config": {"base": 32, "mid": 48, "wide": 64,
                           "res": scenes.BASE,
                           "n_per_class": args.n_per_class, "seed": args.seed}},
               tmp)
    os.replace(tmp, path)


def write_status(step, loss):
    os.makedirs(DATA, exist_ok=True)
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"step": int(step), "loss": (round(float(loss), 4)
                                              if loss is not None else None)}, f)
    os.replace(tmp, STATUS)


def main():
    from model import KhufuNet, count_params, grow_state
    import scenes

    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cpu"
    torch.manual_seed(args.seed)

    print("🎨 خوفو — تجهيز البيانات الإجرائية…", flush=True)
    imgs, labels = scenes.make_dataset(args.n_per_class, seed=args.seed)
    x_all = torch.from_numpy(scenes.to_model_range(imgs)).permute(0, 3, 1, 2).contiguous()  # HWC→CHW
    y_all = torch.from_numpy(labels)
    n = x_all.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    hold = perm[: max(1, n // 20)]                                 # 5% للاختبار
    train_idx = perm[n // 20:]
    print(f"   عينات تدريب: {len(train_idx)} | اختبار: {len(hold)} | "
          f"أصناف: {len(scenes.CLASS_IDS)} | دقة القاعدة: {scenes.BASE}×{scenes.BASE}", flush=True)

    model = KhufuNet().to(device)
    start = 0
    _, saved_ema, prev_step = load_state()
    if args.resume and saved_ema is not None:
        w, _, start = load_state()
        # خ3: لو الأصناف زادت عن آخر تدريب، جدول الأصناف بيتكبر بصفوف جديدة
        # والأصناف القديمة بتحتفظ بكل اللي اتعلمته
        model.load_state_dict(
            grow_state({k: torch.as_tensor(v) for k, v in w.items()},
                       len(scenes.CLASS_IDS)))
        # خ2: EMA بيكمل من آخر حفظ بدل ما يتصفر — الجولات المتتالية (كل 8 ساعات)
        # مش لازم تعيد تسخينه من أول وجديد كل مرة
        ema = {k: v.detach().clone().float()
               for k, v in grow_state({k: torch.as_tensor(v) for k, v in saved_ema.items()},
                                      len(scenes.CLASS_IDS)).items()}
        print(f"↩️  تكملة من الخطوة {start} (EMA محفوظ)", flush=True)
    else:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    betas, alphas, abar = ddpm_tables(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    budget = args.minutes * 60 if args.minutes else None
    total = start + args.steps

    print(f"🧠 معاملات خوفو: {count_params(model):,} | الجهاز: {device} "
          f"| أنوية: {torch.get_num_threads()}", flush=True)
    print("--- بدء التدريب ---", flush=True)

    gen = torch.Generator().manual_seed(args.seed + 1)
    active, last = 0.0, time.time()
    step = start
    run_loss, run_n = 0.0, 0
    last_reported = None
    while step < total:
        now = time.time()
        active += min(now - last, 30.0)
        last = now
        if budget and active >= budget:
            print("⏱️  خلص وقت الجولة — بحفظ", flush=True)
            break
        idx = torch.randint(0, len(train_idx), (args.batch,), generator=gen)
        xb = x_all[train_idx[idx]]
        yb = y_all[train_idx[idx]]
        t = torch.randint(0, T_TOTAL, (args.batch,), generator=gen)
        noise = torch.randn(xb.shape, generator=gen)
        xb_noisy = abar[t][:, None, None, None] * xb \
            + (1 - abar[t])[:, None, None, None].sqrt() * noise
        pred = model(xb_noisy, t, yb)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():                                   # EMA للنموذج
            for k, v in model.state_dict().items():
                ema[k].mul_(0.995).add_(v.detach(), alpha=0.005)
        run_loss += float(loss)
        run_n += 1
        step += 1
        if step % args.eval_every == 0:
            with torch.no_grad():
                hb = hold[: args.batch]
                tn = torch.randint(0, T_TOTAL, (len(hb),), generator=gen)
                hn = torch.randn(x_all[hb].shape, generator=gen)
                hx = abar[tn][:, None, None, None] * x_all[hb] \
                    + (1 - abar[tn])[:, None, None, None].sqrt() * hn
                hloss = float(F.mse_loss(model(hx, tn, y_all[hb]), hn))
            last_reported = {"train": run_loss / run_n, "holdout": hloss}
            print(f"خطوة {step} | خسارة {run_loss / run_n:.4f} "
                  f"| اختبار {hloss:.4f} | {active / 60:.1f} دقيقة", flush=True)
            run_loss, run_n = 0.0, 0
            save_state(CKPT, model, ema, step, args)
            write_status(step, hloss)

    if run_n:
        last_reported = {"train": run_loss / run_n}
    save_state(CKPT, model, ema, step, args)
    write_status(step, (last_reported or {}).get("holdout"))
    print(f"--- انتهى التدريب عند خطوة {step} | آخر خسارة: {last_reported} ---", flush=True)
    print(f"✅ اتحفظ في {CKPT} + {STATUS}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
