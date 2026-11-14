"""
اختبارات سريعة لمكونات seshat.chet — من غير torch

    python tests.py

بتتأكد إن التنضيف والتوكنزر والذاكرة والمحرك والحساب شغالين صح.
لو كل الاختبارات عدت بيطلع "كل الاختبارات عدت" وبيقفل بـ exit code 0.
"""

import os
import sys
import tempfile

FAILURES = []


def check(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


print("1) تنضيف السؤال (tokenizer.normalize_question)")
from tokenizer import CharTokenizer, normalize_question, is_arabic

check("أ إ آ ى ؤ ئ بتتحول", normalize_question("أنا إسميآ ئيزي") == "انا اسميا ييزي")
check("ة بتبقى ه", normalize_question("مدرسة") == "مدرسه")
check("التشكيل بيتشال", normalize_question("مَين بَنى الهرم") == "مين بني الهرم")
check("الأرقام العربية بتبقى إنجليزي", normalize_question("٧ × ٨") == "7 × 8")
check("علامات الاستفهام بتتشال", normalize_question("عامل ايه؟؟") == "عامل ايه")
check("الحروف الكبيرة بتبقى صغيرة", normalize_question("WHAT IS YOUR NAME") == "what is your name")
check("حرف متكرر بيتقل", normalize_question("ازييييك") == "ازيك")
check("is_arabic", is_arabic("مرحبا") and not is_arabic("hello"))

print("\n2) التوكنزر (encode/decode)")
tok = CharTokenizer()
code = "def f():\n    return 1"
check("round-trip للنص العادي", tok.decode(tok.encode(code)) == code)
check("توكن الإزاحة (4 مسافات = توكن واحد)",
      len(tok.encode("    x")) == 2 and tok.decode(tok.encode("    x")) == "    x")
check("حرف غريب بيتحول unk مش بيتمسح", tok.encode("Ω") == [tok.unk_id])
p = tok.encode_prompt("اسمك ايه")
check("شكل الـ prompt", p[0] == tok.user_id and p[-1] == tok.bot_id)

print("\n3) تصليح إزاحة الكود (engine.fix_code_indent)")
from engine import fix_code_indent

bad = "```python\ndef f():\n   return 1\n```"
check("3 مسافات بتتقرب لـ 4", "    return 1" in fix_code_indent(bad))
bad2 = "```javascript\nif (x) {\n    y = 1;\n  }\n```"
check("إزاحة الجافاسكريبت = مسافتين", "  y = 1;" in fix_code_indent(bad2))

print("\n4) ذاكرة الأسئلة (memory.py)")
from memory import QuestionMemory

mem = QuestionMemory()
known, unknown_score = mem.lookup("مين طورك")
check("نفس السؤال بالحرف = 100%", known is not None and known.score >= 0.999)
known, _ = mem.lookup("مين اللي برمجك")
check("صيغة مختلفة بتوصل لنفس السؤال",
      known is not None and known.intent.canonical == "مين طورك")
gib, _ = mem.lookup("asdf qwer zxcv")
check("كلام ملخبط مبيطابقش حاجة", gib is None or gib.score < 0.3)

print("\n5) المحرك: الردود والحساب ورفض اللي مش عايفه")
from engine import Seshat

seshat = Seshat()
r = seshat.reply("اسمك ايه")
check("سؤال الهوية", "seshat" in r["response"] or "عمرو" in r["response"], r["response"])
check("سطر التفكير موجود", r["thinking"].startswith("أقرب سؤال أعرفه"))

r = seshat.reply("7 + 8")
check("حساب صغير", r["response"] == "7 + 8 = 15", r["response"])
r = seshat.reply("154 + 382")
check("حساب بأرقام كبيرة (متدربش عليها)",
      r["response"] == "154 + 382 = 536" and not r["fallback"], r["response"])
r = seshat.reply("12 × 15")
check("ضرب بأرقام كبيرة", r["response"] == "12 × 15 = 180" and not r["fallback"], r["response"])

r = seshat.reply("ايه رايك في سعر الدولار النهارده")
check("سؤال برّه التدريب بيترد عليه بصراحة", r["fallback"] and r["source"] == "none")
r = seshat.reply("asdf qwer zxcv")
check("كلام ملخبط بيرفض", r["fallback"])

r = seshat.reply("اكتب كود بايثون يجمع رقمين")
check("الرد على طلب كود فيه بلوك بايثون", "```python" in r["response"], r["response"][:80])

print("\n6) البيانات والتشيك بوينت")
from dataset import build_intents

intents = build_intents(holdout=True, verbose=False)
check("عدد الردود معقول", len(intents) > 2000, f"{len(intents)} رد")
cats = {it.cat for it in intents}
check("كل الأنواع موجودة", cats == {"chat", "knowledge", "programming", "code",
                                    "translate", "capital", "math", "unknown"})
check("كل Intent له سطر تفكير", all(it.think for it in intents))

from checkpoint import save_checkpoint, load_checkpoint

tmp_npz = os.path.join(tempfile.mkdtemp(), "test_model.npz")
w = {"tok_emb.weight": [[1.0, 2.0], [3.0, 4.0]]}
save_checkpoint(tmp_npz, w, {"n_layer": 1}, ["<pad>", "a"], step=5)
back = load_checkpoint(tmp_npz)
check("حفظ وتحميل التشيك بوينت",
      back["weights"]["tok_emb.weight"][1][0] == 3.0 and back["step"] == 5
      and back["tokens"] == ["<pad>", "a"])

print("\n7) المخ الجديد (brain.py): الأدب والحساب والرسم والتعلم")
import brain as brain_mod
from brain import is_insult, parse_teach, try_math, try_draw, LearnedMemory, Brain

check("شتم: انت اهبل", is_insult("انت اهبل"))
check("شتم بالأرقام: ا7مق", is_insult("ا7مق"))
check("شتم مركب: ابن الكلب", is_insult("ابن الكلب"))
check("شتم إنجليزي: you are stupid", is_insult("you are stupid"))
check("شتم لاتيني: ks emak", is_insult("ks emak"))
check("مش شتم: شكرا يا صديقي", not is_insult("شكرا يا صديقي"))
check("مش شتم: ارسم لي كلب صغير", not is_insult("ارسم لي كلب صغير"))
check("مش شتم: الزبون محتاج مساعدة", not is_insult("الزبون محتاج مساعدة"))
check("شتم موجّه: انت كلب", is_insult("انت كلب"))

check("تعليم سليم", parse_teach("علمني: س: لعبة عمرو ج: الشطرنج") ==
      ("لعبة عمرو", "الشطرنج"))
check("تعليم غلط بيرفض", parse_teach("علمني حاجة كويسة") is None)

check("حساب: 7 × 8", try_math("7 × 8") == "7 × 8 = 56")
check("حساب أرقام كبيرة", try_math("1543 + 2991") == "1543 + 2991 = 4534")
check("حساب بكسور", try_math("12.5 + 3.5") == "12.5 + 3.5 = 16")

d = try_draw("ارسم لي قطة بتنظر للقمر")
check("رسم: بيطلع رابط pollinations من غير params",
      d is not None and d["url"].startswith("https://image.pollinations.ai/prompt/")
      and "?" not in d["url"], str(d))
check("رسم: بيستخرج الوصف", d["prompt"] == "قطة بتنظر للقمر", str(d))
check("مش رسم: يعني ايه رسم", try_draw("يعني ايه رسم") is None)

tmp_dir = tempfile.mkdtemp()
learned = LearnedMemory(os.path.join(tmp_dir, "learned.json"))
learned.add("لعبة عمرو المفضلة", "الشطرنج", src="user")
check("ذاكرة التعلم: نفس السؤال", (learned.lookup("لعبة عمرو المفضلة ايه") or {}).get("a") == "الشطرنج")
check("ذاكرة التعلم: صيغة مختلفة", (learned.lookup("ايه لعبة عمرو المفضلة") or {}).get("a") == "الشطرنج")
check("ذاكرة التعلم: سؤال تاني خالص مبيتخفيش",
      learned.lookup("مين اكتشف بنزين") is None)

class _DummyEngine:
    """محرك وهمي ردوده كلها fallback — عشان نجبر المخ يمشي في الخطوات الجديدة"""
    def reply(self, *a, **kw):
        return {"response": "", "thinking": "", "confidence": 0.0,
                "seconds": 0, "fallback": True, "source": "none",
                "matched": "", "suggestion": ""}

b = Brain(_DummyEngine())
b.learned = LearnedMemory(os.path.join(tmp_dir, "learned2.json"))
b.cfg.api_key = ""                              # من غير مخ خارجي
b.cfg.search_enabled = False                    # ومن غير نت — عشان الاختبار يبقى سريع ومضمون
r = b.reply("انت اهبل")
check("المخ: الشتم بياخد رد مؤدب", r["source"] == "politeness" and "يسامحك" in r["response"])
r = b.reply("7 × 8")
check("المخ: الحساب مضمون", r["source"] == "math" and r["response"] == "7 × 8 = 56")
r = b.reply("ارسم قرد بيغني")
check("المخ: الرسم بيرجع صورة", r["source"] == "draw" and r["image"] and r["image"].startswith("https://image.pollinations.ai/prompt/"))
r = b.reply("علمني: س: ايه هواية عمرو ج: البرمجة")
check("المخ: التعلم اليدوي", r["source"] == "learn")
r = b.reply("ايه هواية عمرو")
check("المخ: بيرد من اللي اتعلمه", r["source"] == "learned" and r["response"] == "البرمجة")
r = b.reply("انسى: ايه هواية عمرو")
check("المخ: النسيان بيمسح", r["source"] == "learn" and "انسيتها" in r["response"])
r = b.reply("ايه هواية عمرو")
check("المخ: بعد النسيان مش عارف", r["fallback"] and r["source"] == "none")
r = b.reply("انسى: سؤال مش موجود خالص")
check("المخ: نسيان حاجة مش موجودة بيقول كده", r["fallback"])
r = b.reply("سؤال غريب خالص عن غيبزة محسمة")
check("المخ: لو مفيش حاجة بيقول الصراحة", r["fallback"] and r["source"] == "none")
r = b.reply("")
check("المخ: رسالة فاضية", "اكتب حاجة" in r["response"])

print("\n" + "—" * 40)
if FAILURES:
    print(f"❌ {len(FAILURES)} اختبار فشل: {', '.join(FAILURES)}")
    sys.exit(1)
print("✅ كل الاختبارات عدت")
