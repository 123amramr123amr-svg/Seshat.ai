"""
قياس نموذج seshat.chet — من غير torch

    python evaluate.py              نسبة الردود الصح بالحرف لكل نوع
    python evaluate.py --samples    يطبع ردود على أسئلة تجريبية
    python evaluate.py --limit 300  يقيس على عينة بس (أسرع)

الرد بيتحسب "صح" لو طلع مطابق بالحرف للرد اللي في بيانات التدريب.
"""

import argparse
import random
import sys
import time

from engine import Seshat, CONFIDENCE_THRESHOLD

SAMPLES = [
    "انت مين", "عامل ايه", "بتعرف تعمل ايه", "مين بنى الهرم الاكبر", "عاصمة مصر ايه",
    "يعني ايه book", "قلم بالانجليزي", "7 × 8", "12 + 9", "يعني ايه ذكاء اصطناعي",
    "اكتب كود بايثون يحسب المضروب", "عايز كود جافاسكريبت يعكس نص",
    "who are you", "what can you do", "what is the capital of japan",
    "what is a neural network", "translate water to arabic", "what is 6 times 7",
    "write python code to check if a number is prime", "how to create a table in sql",
    # أسئلة برّه التدريب — المفروض يقول إنه مش عارف
    "ايه رايك في سعر الدولار", "who won the world cup in 2010", "asdf qwer zxcv",
]


def run_samples(seshat):
    for q in SAMPLES:
        r = seshat.reply(q)
        tag = " [مش عارف]" if r["fallback"] else ""
        print(f"س: {q}\nج: {r['response']}\n   (ثقة {r['confidence']:.0%} — {r['seconds']} ث){tag}\n")


def run_accuracy(seshat, limit, seed):
    # dataset.py بيستورد torch عشان التدريب؛ هنا محتاجين منه البيانات بس
    try:
        from dataset import build_intents
    except ImportError:
        sys.exit("القياس الكامل محتاج torch متثبت (pip install -r requirements-train.txt).\n"
                 "جرب:  python evaluate.py --samples")

    intents = build_intents(holdout=True, verbose=False)
    rng = random.Random(seed)
    seen = [(rng.choice(it.questions), it, "اتدرب عليها") for it in intents if it.questions]
    unseen = [(q, it, "صيغ جديدة") for it in intents for q in it.holdout]
    if limit and len(seen) > limit:
        seen = rng.sample(seen, limit)

    stats = {}
    t0 = time.time()
    for q, it, kind in seen + unseen:
        r = seshat.reply(q, use_fallback=False)
        ok = r["response"] == it.answer
        for key in ((kind, it.cat), (kind, "الكل")):
            s = stats.setdefault(key, [0, 0, 0])
            s[0] += ok
            s[1] += 1
            s[2] += r["confidence"] < CONFIDENCE_THRESHOLD      # هيتقال عليها "مش عارف"

    names = {"chat": "محادثة", "knowledge": "معلومات", "code": "أكواد", "translate": "ترجمة",
             "capital": "عواصم", "math": "حساب", "الكل": "الكل"}
    for kind in ("اتدرب عليها", "صيغ جديدة"):
        print(f"\n— أسئلة {kind} —")
        for (k, cat), (ok, n, low) in sorted(stats.items(), key=lambda x: x[0][1] == "الكل"):
            if k == kind:
                print(f"  {names.get(cat, cat):<8} {ok:>5}/{n:<5} صح = {ok / n * 100:5.1f}%"
                      f"   | ثقة أقل من الحد: {low}")
    print(f"\nالوقت: {time.time() - t0:.0f} ثانية")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    seshat = Seshat()
    print(f"النموذج: {seshat.num_params():,} معامل | خطوات التدريب: {seshat.step} "
          f"| المحرك: {seshat.engine_name}\n")
    if args.samples:
        run_samples(seshat)
    else:
        run_accuracy(seshat, args.limit, args.seed)


if __name__ == "__main__":
    main()
