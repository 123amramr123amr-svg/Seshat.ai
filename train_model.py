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
    p.add_argument("--block-size", type=int, default=4096)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--warmup", type=int, default=-1,
                   help="خطوات التسخين (-1 = تلقائي: خطوات/٢٠ بحد ٢٠٠)")
    p.add_argument("--threads", type=int, default=0, help="عدد أنوية الـ CPU (0 = تلقائي)")
    p.add_argument("--resume", action="store_true", help="يكمل من الملف المحفوظ")
    p.add_argument("--freeze", type=int, default=0,
                   help="خ17: تجميد الـembeddings وأول N طبقات — الطبقات العلوية بس اللي بتتعلم (مقاومة النسيان)")
    p.add_argument("--freeze-blocks", type=int, default=0,
                   help="خ18: تجميد أول N بلوكات بس مع إبقاء الـembeddings قابلة للتدريب — بيصلح خلل خ17: صفوف المفردات الجديدة بتتعلم بدل ما تفضل عشوائية متجمدة")
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
        old_tok = CharTokenizer(ckpt["tokens"])
        old_tok.add_extra_tokens()             # نموذج قديم؟ نضيفله توكنز التفكير والإزاحة
        ds = Dataset(tokenizer=old_tok, holdout=not args.no_holdout,
                     seed=args.seed + ckpt["step"])
        cfg = Config(**ckpt["config"])
        # السياق: بناخد الأكبر بين إعداد الملف والإعداد اللي المستخدم طربه —
        # يعني resume عادي بعد كده بيكبر السياق تلقائيًا لـ 4096 (جدول المواقع
        # بيتوسّع بتكرار الصفوف، شوف تحت). لو عايز تثبّت السياق القديم:
        # --block-size بنفس رقم الملف.
        cfg.block_size = max(cfg.block_size, args.block_size)
        cfg.vocab_size = ds.tok.vocab_size     # ممكن يكون كبر (حروف أو توكنز جديدة)
    else:
        ds = Dataset(holdout=not args.no_holdout, seed=args.seed)
        cfg = Config(block_size=args.block_size, dropout=args.dropout,
                     vocab_size=ds.tok.vocab_size, **PRESETS[args.size])
    tok = ds.tok

    # ---------- النموذج ----------
    model = MiniGPT(cfg).to(device)
    start_step = 0
    vocab_grew = False
    ctx_grew = False
    if ckpt:
        state = {k: torch.from_numpy(np.array(v)) for k, v in ckpt["weights"].items()}
        old_vocab = state["tok_emb.weight"].shape[0]
        vocab_grew = cfg.vocab_size > old_vocab
        if vocab_grew:
            # المفردات كبرت: بنكبّر جدول التضمين ونسيب الصفوف القديمة زي ما هي.
            # خ18: الصفوف الجديدة بتاخد تهيئة دافئة بمتوسط الصفوف القديمة بدل
            # العشوائي — الحروف الجديدة تبدأ من نقطة منطقية في فضاء التضمين
            # وتتعلم أسرع، ولو الـembeddings اتجمّدت تبقى شغالة من أول خطوة
            # بدل ما تفضل قيم عشوائية متجمدة (خلل خ17 الموثق).
            emb = model.tok_emb.weight.detach().clone()
            emb[:old_vocab] = state["tok_emb.weight"]
            emb[old_vocab:] = state["tok_emb.weight"].mean(dim=0)
            space = tok.stoi[" "]
            for t in ("    ", "  "):
                if tok.stoi[t] >= old_vocab:
                    emb[tok.stoi[t]] = state["tok_emb.weight"][space]
            state["tok_emb.weight"] = state["lm_head.weight"] = emb
            print(f"🔤 المفردات كبرت من {old_vocab} لـ {cfg.vocab_size} توكن (تهيئة دافئة للمقاعد الجديدة)")

        # توسيع السياق: جدول المواقع القديم أصغر من السياق الجديد → بنكرر الصفوف
        # (توسيع دوره). المواقع الجديدة بتبدأ من قيم متعلمة، والتدريب على الأمثلة
        # الطويلة (الأغلفة الطويلة في dataset.py) بيظبطها على طول. المواقع القديمة
        # (0..old_ctx-1) بتفضل زي ما هي فجودة الردود القصيرة متبوظش.
        old_ctx = state["pos_emb.weight"].shape[0]
        ctx_grew = cfg.block_size > old_ctx
        if ctx_grew:
            reps = -(-cfg.block_size // old_ctx)          # عدد مرات التكرار
            tiled = state["pos_emb.weight"].repeat(reps, 1)[:cfg.block_size]
            state["pos_emb.weight"] = tiled
            print(f"📏 السياق كبر من {old_ctx} لـ {cfg.block_size} حرف "
                  f"(جدول المواقع اتكرر {reps} مرات — هيتظبط بالتدريب)")

        model.load_state_dict(state)
        start_step = ckpt["step"]
        # خ26 — تحرير أوزان المصدر بعد التحميل: النموذج الواحد بقى 100M+ معامل،
        # والذاكرة الساكنة للتدريب (أوزان + تدرجات + حالات AdamW) محتاجة كل فاضي
        # ممكن. نسخ الأوزان الأصلية (ckpt) والنسخة المؤقتة (state) مالهاش لازمة
        # بعد load_state_dict — النموذج عنده نسخته المستقلة. حررنا ~824MB
        # عشان جلسات الـ100M تعدّي من غير قتل OOM (اتوثق في خ26).
        del state
        ckpt["weights"] = None
        print(f"↩️  تكملة من الخطوة {start_step}")

    # خ17 — تجربة مقاومة النسيان: تجميد أول N طبقات + جدول المواقع. الـembeddings
    # بتتجميد بس لو المفردات ما كبرتش — لو كبرت لازم تتدرب عشان التوكنز الجديدة
    # ما تفضلش معلقة على قيم البداية طول العمر.
    if args.freeze_blocks > 0:
        # خ18 — تجميد البلوكات السفلية فقط: الـembeddings وجدول المواقع والبلوكات
        # العلوية كلها بتتعلم، فصفوف المفردات الجديدة (زي نمو 177→186) بتاخد
        # قيم حقيقية بدل ما تفضل عشوائية متجمدة زي خلل خ17.
        frozen = 0
        for pname, p in model.named_parameters():
            head = pname.split(".")[0]
            idx = int(pname.split(".")[1]) if head == "blocks" else -1
            if head == "blocks" and idx < args.freeze_blocks:
                p.requires_grad_(False)
                frozen += 1
        print(f"🧊 خ18 تجميد بلوكات: أول {args.freeze_blocks} بلوكات بس — "
              f"الـembeddings والمواقع والبلوكات العلوية بتتعلم ({frozen} موتر متجمد)")
    elif args.freeze > 0:
        grew = bool(ckpt and vocab_grew)
        frozen = 0
        for pname, p in model.named_parameters():
            head = pname.split(".")[0]
            idx = int(pname.split(".")[1]) if head == "blocks" else -1
            if head == "pos_emb" or (head == "blocks" and idx < args.freeze):
                p.requires_grad_(False)
                frozen += 1
            elif head in ("tok_emb", "lm_head") and not grew:
                p.requires_grad_(False)
                frozen += 1
        note = " (المفردات كبرت فالتضمين مساب يتدرب)" if grew else ""
        print(f"🧊 تجميد: أول {args.freeze} طبقات + جدول المواقع{note} ({frozen} موتر مش هيتدرب)")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and p.requires_grad]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99))
    # حالة الـ optimizer القديمة مش تنفع لو الجداول كبرت (أشكال جديدة)
    if ckpt and not vocab_grew and not ctx_grew and os.path.exists(opt_path(args.out)):
        opt_state = torch.load(opt_path(args.out), map_location=device, weights_only=True)
        if opt_state.get("step") == start_step:
            opt.load_state_dict(opt_state["optimizer_state_dict"])

    print(f"🧠 المعاملات: {model.num_params():,} | الطبقات: {cfg.n_layer} | الرؤوس: {cfg.n_head} "
          f"| البعد: {cfg.n_embd} | السياق: {cfg.block_size} حرف | المفردات: {cfg.vocab_size}")
    print(f"💻 الجهاز: {device} | أنوية: {torch.get_num_threads()}")

    seen_set, unseen_set = ds.eval_sets()
    total_steps = start_step + args.steps
    warmup = args.warmup if args.warmup >= 0 else min(200, max(1, args.steps // 20))
    budget = args.minutes * 60 if args.minutes else None
    # خ12 — إعداد النافذة المنزلقة بيتحفظ في الملف عشان التشغيل الطويل
    # (> block_size) يستخدم نفس القيم، والتدريب نفسه بيفضل انتباه كامل.
    stream_window = int(getattr(cfg, "stream_window", 768))
    n_sink = int(getattr(cfg, "n_sink", 8))

    def progress(step, elapsed):
        p = (step - start_step) / max(1, args.steps)
        if budget:
            p = max(p, elapsed / budget)
        return min(1.0, p)

    def lr_at(step, elapsed):
        # الـ warmup شغال كمان في التكملة: أول خطوات الجولة بـ lr صغير عشان
        # ما نصدمش النموذج المتقارب (جودة الردود الحرفية حساسة جدًا للحركة)
        if step - start_step < warmup:
            return args.lr * (step - start_step + 1) / warmup
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress(step, elapsed))))

    # ---------- التدريب ----------
    print("\n--- بدء التدريب ---")
    model.train()
    # كل خانة في مصفوفة الانتباه بتاخد حوالي 24 بايت وقت التدريب (لكل رأس ولكل طبقة)
    # — خ12: مع النافذة المنزلقة العرض الفعلي بيتبلى min(T, window+n_sink)
    max_cells = int(args.ram_gb * 1e9 / 24 / (cfg.n_head * cfg.n_layer))
    batches = ds.batches(args.batch, cfg.block_size, device, max_cells=max_cells)
    # بنحسب وقت التدريب الفعلي بس: لو الجهاز نام أو اتوقف شوية، الوقت ده مبيتحسبش
    active, last = 0.0, time.time()
    run_loss, run_n = 0.0, 0
    metrics = {}
    step = start_step
    try:
        while step < total_steps:
            now = time.time()
            active += min(now - last, 30.0)
            last = now
            elapsed = active
            if budget and elapsed >= budget:
                print("⏱️  خلص الوقت المحدد")
                break
            lr = lr_at(step, elapsed)
            for g in opt.param_groups:
                g["lr"] = lr

            xb, yb = next(batches)
            # خ12: المستندات الطويلة بتتدرب بانتباه كامل (زي أي مثال) — الحقيقة
            # بتقد تفصيلة في أي مكان في المستند. النافذة المنزلقة في التشغيل
            # بتستخدم بس للنصوص الأطول من block_size (شوف engine.reply_long) —
            # دي نفس معمارية StreamingLLM: تدريب كامل + تدفق بالنافذة وقت التشغيل.
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
                           "minutes": active / 60}
                print(f"خطوة {step:5d} | خسارة {run_loss / run_n:.4f} | lr {lr:.5f} "
                      f"| ردود صح (اتشافت) {seen_acc * 100:5.1f}% "
                      f"| ردود صح (صيغ جديدة) {un_acc * 100:5.1f}% "
                      f"| {active / 60:5.1f} دقيقة", flush=True)
                run_loss, run_n = 0.0, 0
                save(args.out, model, cfg, tok, step, metrics, opt)
                trim_memory()
    except KeyboardInterrupt:
        print("\n⛔ اتوقف بالإيد — بحفظ اللي اتعلمه لحد دلوقتي")

    seen_loss, seen_acc = ds.evaluate(model, seen_set, device)
    un_loss, un_acc = ds.evaluate(model, unseen_set, device)
    metrics = {"seen_exact": seen_acc, "unseen_exact": un_acc, "unseen_loss": un_loss,
               "minutes": active / 60}
    save(args.out, model, cfg, tok, step, metrics, opt)
    print("--- انتهى التدريب ---\n")
    print(f"✅ اتحفظ في {args.out}")
    print(f"   ردود صح بالحرف على أسئلة اتدرب عليها: {seen_acc * 100:.1f}%")
    print(f"   ردود صح بالحرف على صيغ ماشافهاش:      {un_acc * 100:.1f}%  ({len(unseen_set)} سؤال)")
    print("\nشغّل دلوقتي:  python app.py")


if __name__ == "__main__":
    main()
