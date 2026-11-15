"""
قياس نموذج seshat.chet — من غير torch

    python evaluate.py               نسبة الردود الصح بالحرف لكل نوع (النموذج لوحده)
    python evaluate.py --paraphrase  أسئلة بصيغ جديدة خالص: النموذج لوحده × النموذج + الذاكرة
    python evaluate.py --draw        الرسم: تركيبات اتدرب عليها × تركيبات (شكل + لون) عمره ما شافها
    python evaluate.py --understand  فهم الأسئلة اللي برّه معلوماته، على مواضيع عمره ما شافها
    python evaluate.py --samples     يطبع ردود على أسئلة تجريبية
    python evaluate.py --limit 300   يقيس على عينة بس (أسرع)

الرد بيتحسب "صح" لو طلع مطابق بالحرف للرد اللي في بيانات التدريب.
"""

import argparse
import random
import sys
import time

from engine import Seshat, CONFIDENCE_THRESHOLD
from tokenizer import normalize_question

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
        think = f"🧠 {r['thinking']}\n" if r["thinking"] else ""
        print(f"س: {q}\n{think}ج: {r['response']}\n"
              f"   (ثقة {r['confidence']:.0%} — {r['seconds']} ث){tag}\n")


def run_paraphrase(seshat, show):
    """
    أسئلة data/paraphrase_tests.py مكتوبة بصيغ مش موجودة في البيانات.
    بنقارن النموذج لوحده بالنموذج + الذاكرة. الذاكرة هنا من غير الصيغ اللي
    متشالة للاختبار، عشان المقارنة تفضل عادلة.
    """
    from dataset import DATA_DIR, build_intents
    from memory import QuestionMemory
    sys.path.insert(0, DATA_DIR)
    import paraphrase_tests as pt
    import unknown_questions as uq

    intents = build_intents(holdout=True, verbose=False)
    answers = {}
    for it in intents:
        answers.setdefault(normalize_question(it.canonical), it.answer)
    seshat.memory = QuestionMemory(intents, include_holdout=False)

    inside = [(q, {answers[normalize_question(c)] for c in ok},
               {normalize_question(c) for c in ok}) for q, ok in pt.IN_SCOPE]
    outside = pt.OUT_OF_SCOPE + uq.AR_TEST + uq.EN_TEST
    print(f"{len(inside)} سؤال بصيغة جديدة يعرف ردها | {len(outside)} سؤال برّه اللي اتعلمه\n")

    for name, use_memory in (("النموذج لوحده", False), ("النموذج + الذاكرة", True)):
        right = wrong = silent = hinted = 0
        misses = []
        for q, ok, ok_questions in inside:
            r = seshat.reply(q, use_memory=use_memory)
            if r["fallback"]:
                silent += 1
                good_hint = normalize_question(r.get("suggestion", "")) in ok_questions
                hinted += good_hint
                misses.append(f"    مش عارف ← {q}" + ("   (واقترح السؤال الصح)" if good_hint else ""))
            elif r["response"] in ok:
                right += 1
            else:
                wrong += 1
                misses.append(f"    رد غلط ← {q}   ({r['thinking']})")
        refused = made_up = 0
        for q in outside:
            r = seshat.reply(q, use_memory=use_memory)
            if r["fallback"]:
                refused += 1
            else:
                made_up += 1
                misses.append(f"    ألّف رد ← {q}   ({r['thinking']})")
        n = len(inside)
        print(f"— {name} —")
        print(f"  صيغ جديدة:      رد صح {right}/{n} = {right / n:.0%} | رد غلط {wrong} | قال مش عارف {silent}"
              + (f" (منهم {hinted} اقترح فيهم السؤال الصح)" if use_memory else ""))
        print(f"  برّه اللي اتعلمه: قال مش عارف {refused}/{len(outside)} = "
              f"{refused / len(outside):.0%} | رد برد غلط {made_up}")
        if show:
            print("\n".join(misses))
        print()


def run_draw(seshat, show, limit=0):
    """
    الرسم بيتقاس من الطريق الكامل (reply): النموذج بيكتب الرسمة والمحرك بيراجعها.
      • اتدرب عليها: (رسمة، لون) موجودين في التدريب.
      • تركيبات جديدة: (رسمة، لون) متشالين من التدريب (drawings.HELD_OUT) — النموذج شاف
        الرسمة بألوان تانية وشاف اللون على رسومات تانية، بس عمره ما شاف الاتنين مع بعض.
      • مشاهد: رسمتين مع بعض في طلب واحد (عمره ما اتدرب على طلب فيه رسمتين).
    "مطابقة" = كل مربع في مكانه. "اتعرضت" = الغلط مش أكتر من DRAW_MAX_WRONG مربع.
    """
    from engine import drawings, DRAW_MAX_WRONG
    rng = random.Random(11)
    cells = drawings.N * drawings.N
    t0 = time.time()

    def score(name, tests):
        n = exact = shown = wrong = 0
        for q, want in tests:
            r = seshat.reply(q)
            got = drawings.read_full(r["response"]) if not r["fallback"] else None
            n += 1
            if got is not None:
                shown += 1
                bad = sum(a != b for x, y in zip(got, want) for a, b in zip(x, y))
                bad += abs(len(got) - len(want)) * len(want[0])
                wrong += bad
                exact += bad == 0
            if show and (got is None or got != want):
                print(f"  ✗ {q!r} → " + ("اترفضت" if got is None else "اتعرضت وفيها غلط"))
                if got is not None:
                    print("\n".join("      " + a + "   " + b for a, b in zip(got, want)))
        right = 100 - wrong / max(1, shown * len(tests[0][1]) * len(tests[0][1][0])) * 100
        print(f"  {name:<16} مطابقة بالمربع {exact}/{n} = {exact / n * 100:.1f}% | "
              f"اتعرضت {shown}/{n} = {shown / n * 100:.1f}% | مربعات صح في المعروض {right:.1f}%")

    for name, held in (("اتدرب عليها", False), ("تركيبات جديدة", True)):
        tests = []
        for key, code in drawings.combos(held):
            ar_qs, en_qs = drawings._questions(key, code)
            want = drawings.grid_for(key, code)
            tests += [(rng.choice(ar_qs), want), (rng.choice(en_qs), want)]
        if limit and len(tests) > limit:
            tests = rng.sample(tests, limit)
        score(name, tests)

    keys = list(drawings.OBJECTS)
    scenes = []
    for _ in range(min(limit or 40, 40)):
        (k1, k2) = rng.sample(keys, 2)
        c1, c2 = (rng.choice(drawings.COLORS)[0] if k not in drawings.FIXED_COLOR
                  else drawings.OBJECTS[k][6] for k in (k1, k2))
        arabic = rng.random() < 0.6
        t1, t2 = drawings.title(k1, c1, arabic), drawings.title(k2, c2, arabic)
        q = f"ارسم {t1} و{t2}" if arabic else f"draw a {t1} and a {t2}"
        want = drawings.read_full(drawings.full_block([drawings.grid_for(k1, c1),
                                                       drawings.grid_for(k2, c2)]))
        scenes.append((q, want))
    score("مشاهد (رسمتين)", scenes)
    print(f"  (الرسمة = {cells} مربع، وبتتعرض لو الغلط ≤ {DRAW_MAX_WRONG})")
    print(f"الوقت: {time.time() - t0:.0f} ثانية")


def run_understand(seshat, limit, show):
    """
    فهم الأسئلة اللي برّه معلوماته، على مواضيع النموذج عمره ما شافها (understanding.HELD_OUT_*).
    بنقيس حاجتين:
      • النموذج لوحده: كتب النوع والموضوع صح؟ وكتب الجملة كلها صح حرف بحرف؟
      • المحرك كله (النموذج + مراجعة + قواعد): النوع والموضوع اللي طلعوا في الآخر صح؟
    """
    from engine import understanding as un
    items = un.build(held_out=True)
    rng = random.Random(7)
    tests = [(q, kind, topic, arabic) for kind, topic, arabic, qs, _ in items
             for q in (qs[0], qs[len(qs) // 2])]
    if limit and len(tests) > limit:
        tests = rng.sample(tests, limit)
    n = think_ok = text_ok = final_ok = 0
    by = {}
    t0 = time.time()
    for q, kind, topic, arabic in tests:
        q = normalize_question(q)
        alone = seshat.reply(q, use_fallback=False, use_memory=False)
        full = seshat.reply(q)
        n += 1
        t_ok = alone["thinking"] == un.think(kind, topic, arabic)
        x_ok = alone["response"] == un.compose(kind, topic, arabic)
        f_ok = (full.get("kind"), full.get("topic")) == (kind, topic)
        think_ok += t_ok
        text_ok += t_ok and x_ok
        final_ok += f_ok
        by[full.get("understood_by", "-")] = by.get(full.get("understood_by", "-"), 0) + 1
        if show and not (t_ok and x_ok):
            print(f"  ✗ {q!r}\n      🧠 {alone['thinking']!r}\n      ج {alone['response'][:110]!r}")
    print(f"  أسئلة بمواضيع جديدة: {n}")
    print(f"  النموذج لوحده — النوع والموضوع صح: {think_ok}/{n} = {think_ok / n * 100:.1f}%")
    print(f"  النموذج لوحده — الجملة كلها صح حرف بحرف: {text_ok}/{n} = {text_ok / n * 100:.1f}%")
    print(f"  المحرك كله — النوع والموضوع صح: {final_ok}/{n} = {final_ok / n * 100:.1f}%")
    print(f"  الفهم جه منين: {by}")
    print(f"الوقت: {time.time() - t0:.0f} ثانية")


def run_accuracy(seshat, limit, seed):
    from dataset import build_intents

    intents = build_intents(holdout=True, verbose=False)
    rng = random.Random(seed)
    seen = [(rng.choice(it.questions), it, "اتدرب عليها") for it in intents
            if it.questions and it.cat != "unknown"]
    unknown = [it for it in intents if it.cat == "unknown"]      # كتير → بناخد منهم عينة
    seen += [(rng.choice(it.questions), it, "اتدرب عليها")
             for it in rng.sample(unknown, min(150, len(unknown)))]
    unseen = [(q, it, "صيغ جديدة") for it in intents for q in it.holdout]
    if limit and len(seen) > limit:
        seen = rng.sample(seen, limit)

    stats = {}
    t0 = time.time()
    for q, it, kind in seen + unseen:
        r = seshat.reply(q, use_fallback=False, use_memory=False)     # النموذج لوحده
        ok = r["response"] == it.answer
        low = r["confidence"] < CONFIDENCE_THRESHOLD and not r["fallback"]
        for key in ((kind, it.cat), (kind, "الكل")):
            s = stats.setdefault(key, [0, 0, 0, 0, 0])
            s[0] += ok
            s[1] += 1
            s[2] += r["thinking"] == it.think                  # فهم السؤال صح
            s[3] += low and ok                                 # رد صح بس هيترفض بسبب الثقة
            s[4] += (not ok) and (low or r["fallback"])        # رد غلط بس المستخدم هيشوف "مش عارف"

    names = {"chat": "محادثة", "knowledge": "معلومات", "programming": "برمجة", "code": "أكواد",
             "translate": "ترجمة", "capital": "عواصم", "math": "حساب", "draw": "رسم",
             "unknown": "برّه التدريب", "الكل": "الكل"}
    for kind in ("اتدرب عليها", "صيغ جديدة"):
        print(f"\n— أسئلة {kind} —")
        print("  (رد صح بالحرف | فهم السؤال صح | صح واترفض بالثقة | غلط واتقال مش عارف)")
        for (k, cat), (ok, n, th, lost, caught) in sorted(stats.items(),
                                                          key=lambda x: x[0][1] == "الكل"):
            if k == kind:
                print(f"  {names.get(cat, cat):<12} {ok:>5}/{n:<5} = {ok / n * 100:5.1f}%"
                      f" | فهم {th / n * 100:5.1f}% | {lost:>3} | {caught:>3}")
    print(f"\nالوقت: {time.time() - t0:.0f} ثانية")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", action="store_true")
    p.add_argument("--paraphrase", action="store_true")
    p.add_argument("--draw", action="store_true")
    p.add_argument("--understand", action="store_true")
    p.add_argument("--model", default=None, help="ملف نموذج تاني غير الافتراضي")
    p.add_argument("--show", action="store_true", help="مع --paraphrase أو --draw: اطبع اللي غلط فيه")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    seshat = Seshat(args.model) if args.model else Seshat()
    print(f"النموذج: {seshat.num_params():,} معامل | خطوات التدريب: {seshat.step} "
          f"| المحرك: {seshat.engine_name}\n")
    if args.samples:
        run_samples(seshat)
    elif args.paraphrase:
        run_paraphrase(seshat, args.show)
    elif args.draw:
        run_draw(seshat, args.show, args.limit)
    elif args.understand:
        run_understand(seshat, args.limit, args.show)
    else:
        run_accuracy(seshat, args.limit, args.seed)


if __name__ == "__main__":
    main()
