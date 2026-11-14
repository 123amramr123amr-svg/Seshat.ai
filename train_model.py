"""
تدريب نموذج seshat.chet وحفظه على القرص

أمثلة:
    python train_model.py                        # حجم small لمدة 4000 خطوة
    python train_model.py --size tiny --minutes 20
    python train_model.py --size medium --steps 15000      # لو عندك GPU
    python train_model.py --resume               # يكمل تدريب من آخر حفظ
    python train_model.py --no-holdout           # يدرب على كل الصيغ (من غير اختبار)

التدريب بيحفظ تلقائي كل شوية في seshat_chet_model.npz، فلو وقفته بـ Ctrl+C مفيش حاجة هتضيع.
الملف ده هو نفسه اللي بيشتغل بيه السيرفر (من غير torch).
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from checkpoint import load_checkpoint, save_checkpoint
from dataset import Dataset
from model import Config, MiniGPT, PRESETS
from tokenizer import CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="تدريب seshat.chet")
    p.add_argument("--size", default="small", choices=list(PRESETS), help="حجم النموذج")
    p.add_argument("--steps", type=int, default=4000, help="عدد خطوات التدريب")
    p.add_argument("--minutes", type=float, default=0, help="حد أقصى للوقت بالدقايق (0 = من غير حد)")
    p.add_argument("--batch", type=int, default=48, help="أقصى عدد أمثلة في الباتش")
    p.add_argument("--ram-gb", type=float, default=2.0,
                   help="الرام المتاحة للتدريب تقريبًا (بتحدد حجم باتشات الأكواد الطويلة)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--threads", type=int, default=0, help="عدد أنوية الـ CPU (0 = تلقائي)")
    p.add_argument("--resume", action="store_true", help="يكمل من الملف المحفوظ")
    p.add_argument("--no-holdout", action="store_true", help="يدرب على كل صيغ الأسئلة")
    p.add_argument("--out", default=os.path.join(HERE, "seshat_chet_model.npz"))
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def trim_memory():
    """بيرجّع الذاكرة الفاضية للنظام (على لينكس) عشان استهلاك الرام ميزيدش مع الوقت"""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def opt_path(model_path):
    """حالة الـ optimizer بتتحفظ في ملف لوحدها (بتلزم لتكملة التدريب بس، ومش بتترفع)"""
    return os.path.splitext(model_path)[0] + ".optimizer.pt"


def save(path, model, cfg, tok, step, metrics, opt=None):
    weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    save_checkpoint(path, weights, cfg.to_dict(), tok.tokens, step, metrics)
    if opt is not None:
        tmp = opt_path(path) + ".tmp"
        torch.save({"step": step, "optimizer_state_dict": opt.state_dict()}, tmp)
        os.replace(tmp, opt_path(path))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- البيانات ----------
    print("📚 تحميل البيانات:")
    ckpt = None
    if args.resume and os.path.exists(args.out):
        ckpt = load_checkpoint(args.out)
        ds = Dataset(tokenizer=CharTokenizer(ckpt["tokens"]), holdout=not args.no_holdout,
                     seed=args.seed + ckpt["step"])
        cfg = Config(**ckpt["config"])
    else:
        ds = Dataset(holdout=not args.no_holdout, seed=args.seed)
        cfg = Config(block_size=args.block_size, dropout=args.dropout,
                     vocab_size=ds.tok.vocab_size, **PRESETS[args.size])
    tok = ds.tok

    # ---------- النموذج ----------
    model = MiniGPT(cfg).to(device)
    start_step = 0
    if ckpt:
        model.load_state_dict({k: torch.from_numpy(np.array(v))
                               for k, v in ckpt["weights"].items()})
        start_step = ckpt["step"]
        print(f"↩️  تكملة من الخطوة {start_step}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99))
    if ckpt and os.path.exists(opt_path(args.out)):
        opt_state = torch.load(opt_path(args.out), map_location=device, weights_only=True)
        if opt_state.get("step") == start_step:
            opt.load_state_dict(opt_state["optimizer_state_dict"])

    print(f"🧠 المعاملات: {model.num_params():,} | الطبقات: {cfg.n_layer} | الرؤوس: {cfg.n_head} "
          f"| البعد: {cfg.n_embd} | السياق: {cfg.block_size} حرف | المفردات: {cfg.vocab_size}")
    print(f"💻 الجهاز: {device} | أنوية: {torch.get_num_threads()}")

    seen_set, unseen_set = ds.eval_sets()
    total_steps = start_step + args.steps
    warmup = min(200, max(1, args.steps // 20))
    budget = args.minutes * 60 if args.minutes else None

    def progress(step, elapsed):
        p = (step - start_step) / max(1, args.steps)
        if budget:
            p = max(p, elapsed / budget)
        return min(1.0, p)

    def lr_at(step, elapsed):
        if step - start_step < warmup and not ckpt:
            return args.lr * (step - start_step + 1) / warmup
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress(step, elapsed))))

    # ---------- التدريب ----------
    print("\n--- بدء التدريب ---")
    model.train()
    # كل خانة في مصفوفة الانتباه بتاخد حوالي 24 بايت وقت التدريب (لكل رأس ولكل طبقة)
    max_cells = int(args.ram_gb * 1e9 / 24 / (cfg.n_head * cfg.n_layer))
    batches = ds.batches(args.batch, cfg.block_size, device, max_cells=max_cells)
    t0 = time.time()
    run_loss, run_n = 0.0, 0
    metrics = {}
    step = start_step
    try:
        while step < total_steps:
            elapsed = time.time() - t0
            if budget and elapsed >= budget:
                print("⏱️  خلص الوقت المحدد")
                break
            lr = lr_at(step, elapsed)
            for g in opt.param_groups:
                g["lr"] = lr

            xb, yb = next(batches)
            _, loss = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += loss.item()
            run_n += 1
            step += 1

            if step % args.eval_every == 0 or step == total_steps:
                seen_loss, seen_acc = ds.evaluate(model, seen_set, device)
                un_loss, un_acc = ds.evaluate(model, unseen_set, device)
                metrics = {"train_loss": run_loss / run_n, "seen_exact": seen_acc,
                           "unseen_exact": un_acc, "unseen_loss": un_loss,
                           "minutes": (time.time() - t0) / 60}
                print(f"خطوة {step:5d} | خسارة {run_loss / run_n:.4f} | lr {lr:.5f} "
                      f"| ردود صح (اتشافت) {seen_acc * 100:5.1f}% "
                      f"| ردود صح (صيغ جديدة) {un_acc * 100:5.1f}% "
                      f"| {(time.time() - t0) / 60:5.1f} دقيقة", flush=True)
                run_loss, run_n = 0.0, 0
                save(args.out, model, cfg, tok, step, metrics, opt)
                trim_memory()
    except KeyboardInterrupt:
        print("\n⛔ اتوقف بالإيد — بحفظ اللي اتعلمه لحد دلوقتي")

    seen_loss, seen_acc = ds.evaluate(model, seen_set, device)
    un_loss, un_acc = ds.evaluate(model, unseen_set, device)
    metrics = {"seen_exact": seen_acc, "unseen_exact": un_acc, "unseen_loss": un_loss,
               "minutes": (time.time() - t0) / 60}
    save(args.out, model, cfg, tok, step, metrics, opt)
    print("--- انتهى التدريب ---\n")
    print(f"✅ اتحفظ في {args.out}")
    print(f"   ردود صح بالحرف على أسئلة اتدرب عليها: {seen_acc * 100:.1f}%")
    print(f"   ردود صح بالحرف على صيغ ماشافهاش:      {un_acc * 100:.1f}%  ({len(unseen_set)} سؤال)")
    print("\nشغّل دلوقتي:  python app.py")


if __name__ == "__main__":
    main()
