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
check("سطر التفكير موجود", r["thinking"].startswith("فهمت إنك بتسأل"))

r = seshat.reply("7 + 8")
check("حساب صغير", r["response"] == "7 + 8 = 15", r["response"])
r = seshat.reply("154 + 382")
check("حساب بأرقام كبيرة (متدربش عليها)",
      r["response"] == "154 + 382 = 536" and not r["fallback"], r["response"])
r = seshat.reply("12 × 15")
check("ضرب بأرقام كبيرة", r["response"] == "12 × 15 = 180" and not r["fallback"], r["response"])

r = seshat.reply("ايه رايك في سعر الدولار النهارده")
check("سؤال برّه معلوماته: بيفهم نوعه وموضوعه ويكوّن رد",
      r["fallback"] and r["source"] == "understand" and r["kind"] == "opinion"
      and r["topic"] in r["response"], str(r))
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
                                    "translate", "capital", "math", "draw", "unknown",
                                    "longdoc"})
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

# الصيغة المرنة: «علمني: سؤال؟ جواب» من غير س:/ج:
check("تعليم مرن: سؤال وجواب", parse_teach("علمني: ايه هي عاصمة تشاد؟ هي ندجامنا") ==
      ("ايه هي عاصمة تشاد؟", "ندجامنا"))
check("تعليم مرن: سطرين", parse_teach("علمني: ايه عاصمة مالي؟\nهي باماكو") ==
      ("ايه عاصمة مالي؟", "باماكو"))
check("تعليم مرن: شرطة وكلمة سؤال", parse_teach("علمني: مين اخترع المصباح؟ - توماس اديسون") ==
      ("مين اخترع المصباح؟", "توماس اديسون"))
check("تعليم مرن: إنجليزي", parse_teach("teach: what is 2+2? it is 4") ==
      ("what is 2+2?", "it is 4"))
check("تعليم مرن: من غير علامة استفهام ولا كلمة سؤال بيرفض",
      parse_teach("علمني: عاصمة مصر القاهرة") is None)
check("تعليم مرن: من غير جواب بيرفض", parse_teach("علمني: ايه رأيك في الكود؟") is None)
check("تعليم مرن: جواب طويل بيرفض", parse_teach("علمني: ايه ده؟ " + "كلمة " * 120) is None)

check("حساب: 7 × 8", try_math("7 × 8") == "7 × 8 = 56")
check("حساب أرقام كبيرة", try_math("1543 + 2991") == "1543 + 2991 = 4534")
check("حساب بكسور", try_math("12.5 + 3.5") == "12.5 + 3.5 = 16")

# الحساب بالكلمات العربي: في/على/زائد/ناقص/مضروب في/تقسيم على
check("حساب بالكلمات: احسب 17 في 24", try_math("احسب 17 في 24") == "17 × 24 = 408")
check("حساب بالكلمات: 15 على 3", try_math("15 على 3") == "15 ÷ 3 = 5")
check("حساب بالكلمات: 8 زائد 2", try_math("8 زائد 2") == "8 + 2 = 10")
check("حساب بالكلمات: 10 ناقص 4", try_math("10 ناقص 4") == "10 - 4 = 6")
check("حساب بالكلمات: 9 مضروب في 7", try_math("9 مضروب في 7") == "9 × 7 = 63")
# خ8: مسائل متسلسلة بأولوية عمليات
check("خ8 متسلسلة: 2 زائد 3 زائد 10", try_math("2 زائد 3 زائد 10") == "2 + 3 + 10 = 15")
check("خ8 متسلسلة: 5 ناقص 2 مضروب في 4", try_math("5 ناقص 2 مضروب في 4") == "5 - 2 * 4 = -3")
check("خ8 متسلسلة: 10 على 2 على 5", try_math("10 على 2 على 5") == "10 / 2 / 5 = 1")
check("خ8 متسلسلة: احسب 4 زائد 6 على 2", try_math("احسب 4 زائد 6 على 2") == "4 + 6 / 2 = 7")
check("خ8 متسلسلة بالرموز: 5 + 3 * 2", try_math("5 + 3 * 2") == "5 + 3 * 2 = 11")
# خ9: كلمات الأرقام العربي (فصحى + عامية) بتتحول لأرقام قبل الحساب
check("خ9 أرقام بالكلمات: اثنين زائد تلاتة", try_math("اثنين زائد تلاتة") == "2 + 3 = 5")
check("خ9 أرقام بالكلمات: اثنين في خمسة", try_math("اثنين في خمسة") == "2 * 5 = 10")
check("خ9 أرقام بالكلمات: خمسة ناقص اتنين", try_math("خمسة ناقص اتنين") == "5 - 2 = 3")
check("خ9 أرقام بالكلمات: عشرة تقسيم على اتنين", try_math("عشرة تقسيم على اتنين") == "10 / 2 = 5")
check("خ9 تركيب واحد وعشرين: 21 + 3", try_math("واحد وعشرين زائد تلاتة") == "21 + 3 = 24")
check("خ9 تركيب خمسة ومية: 105 + 20", try_math("خمسة ومية زائد عشرين") == "105 + 20 = 125")
check("خ9 مزج أرقام وكلمات: 8 زائد تلاتة", try_math("8 زائد تلاتة") == "8 + 3 = 11")
check("خ9 حماية: جملة فيها أرقام كلام ما تتحسبش", try_math("في 2024 حصلت حاجة كبيرة") is None)
check("خ8 حماية: جملة طويلة فيها «في» ما تتحسبش", try_math("الموضوع في التفاصيل دي كبير في 5 مواضع") is None)
check("حساب بالكلمات: 100 تقسيم على 4", try_math("100 تقسيم على 4") == "100 ÷ 4 = 25")
check("حساب بالكلمات: كم 12 في 12", try_math("كم 12 في 12") == "12 × 12 = 144")
check("جملة عادية فيها «في» ما تتحسبش: في 2024 حصلت حاجة في 3 شهور",
      try_math("في 2024 حصلت حاجة في 3 شهور") is None)
check("جملة عادية: عام 1945 انتهت الحرب في 9 مايو",
      try_math("عام 1945 انتهت الحرب في 9 مايو") is None)

d = try_draw("ارسم لي قطة بتنظر للقمر")
check("صور: بيستخرج الوصف", d is not None and d["prompt"] == "قطة بتنظر للقمر" and d["strong"]
      and not d["pixels"], str(d))
check("صور: اعمل صورة / انشئ صورة / صورة لـ",
      try_draw("اعمل صورة عربية حمرا على البحر")["prompt"] == "عربية حمرا على البحر"
      and try_draw("انشئ صورة لمدينة في المستقبل")["prompt"] == "مدينة في المستقبل"
      and try_draw("صورة للقطة")["prompt"] == "القطة"
      and try_draw("اعمل صورة لمبة منورة")["prompt"] == "لمبة منورة")
check("صور: إنجليزي", try_draw("please draw a castle on a hill")["prompt"] == "castle on a hill"
      and try_draw("generate an image of a dragon")["prompt"] == "dragon")
check("صور: «بالمربعات» = رسم seshat نفسه",
      try_draw("ارسم بالمربعات قطة سودا وشمس") == {"prompt": "قطة سودا وشمس", "strong": True,
                                                     "pixels": True})
check("مش طلب صورة: كلام عن الرسم والصور", all(try_draw(m) is None for m in (
    "بتعرف ترسم", "مين رسم الموناليزا", "ايه في الصورة دي", "الصورة حلوة", "الرسم فن جميل",
    "بعتلك صورة امبارح", "drawing is fun")))
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
r = b.reply("2 زائد 3 زائد 10")
check("المخ: الحساب المتسلسل مضمون (خ8)", r["source"] == "math" and r["response"] == "2 + 3 + 10 = 15", r["response"])

# --- الهوية النصية (خ10): «انت مين» بيرجع تعريف النموذج النصي — بعد حذف نموذج الصور ---
r = b.reply("انت مين")
check("خ10 الهوية: «انت مين» = نموذج نصي من غير صور",
      r["source"] == "identity" and "seshat" in r["response"]
      and "مفيش توليد صور" in r["response"], r["response"])
r = b.reply("who are you?")
check("خ10 الهوية: إنجليزي من غير صور", r["source"] == "identity"
      and "no image generation" in r["response"], r["response"])
r = b.reply("مين اخترع الراديو")
check("الهوية: مش هيدخ سؤال معلومات في الهوية", r["source"] != "identity", r["source"])

# --- خ10: نموذج الصور اتشال خالص — أي طلب صورة بيرجع رد نصي من غير ملف صور ---
r = b.reply("ارسم قرد بيغني")
check("خ10: «ارسم» بيرجع رد نصي صريح — مش صورة",
      r["source"] == "text" and not r["image"] and r["image_id"] is None
      and "مش بعمل صور" in r["response"] and "بالمربعات" in r["response"], str(r))
r = b.reply("اعمل صورة عربية حمرا")
check("خ10: «اعمل صورة» نفس السلوك النصي",
      r["source"] == "text" and not r["image"] and "مش بعمل صور" in r["response"], str(r))
r = b.reply("please draw a dragon")
check("خ10: طلب صورة إنجليزي → رد نصي إنجليزي",
      r["source"] == "text" and "don't make images" in r["response"], str(r))
check("خ10: مفيش ملفات توليد صور في المستودع",
      not os.path.exists("imagegen.py") and not os.path.exists("seshat_unified.py")
      and not os.path.exists("khufu"), "اتأكد إن ملفات الصور مازالت محذوفة")
check("خ10: الحالة بتقول صراحة إن الصور مقفولة",
      b.status()["images"] is False, str(b.status()))
r = b.reply("ارسم بالمربعات قرد بيغني")
check("خ10: «ارسم بالمربعات» لسه شغال — رسم نصي مش صورة",
      r["source"] == "draw" and not r["image"], str(r))
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

print("\n8) فهم الأسئلة (data/understanding.py)")
from engine import understanding as un, drawings

check("كشف رأي + موضوع", un.detect("ايه رايك في القهوه") == ("opinion", "القهوه"))
check("كشف شخص بالإنجليزي", un.detect("who is marie curie") == ("person", "marie curie"))
check("سؤال عادي مش بيتكشف", un.detect("عامل ايه") is None)
check("الرد فيه الموضوع", "القهوه" in un.compose("opinion", "القهوه", True))
check("سطر التفكير بيتقرا تاني",
      un.parse_think(un.think("price", "الدهب", True)) == ("price", "الدهب", True))
check("نوع من غير موضوع", un.parse_think(un.think("future", "", False)) == ("future", "", False))
check("مواضيع الاختبار مش في التدريب",
      not {i[1] for i in un.build()} & {i[1] for i in un.build(True)})
r = seshat.reply("افتحلي يوتيوب")
check("طلب تنفيذ بيتفهم", r["source"] == "understand" and r["kind"] == "action", str(r))
r = seshat.reply("who is marie curie")
check("سؤال عن شخص: قابل للبحث", r["kind"] == "person" and r["searchable"], str(r))

check("سعر: كلمة الوقت مش من الموضوع", un.detect("سعر الدهب النهارده كام") == ("price", "الدهب"))
check("مين اخترع… مش سؤال عن شخص", un.detect("مين اخترع التليفون") is None)
check("كلمة اتبدلت = موضوع تاني", un.other_subject("كم عدد سكان الصين", "عدد سكان مصر"))
check("صيغة تانية لنفس السؤال مش موضوع تاني",
      not un.other_subject("مين الشخص اللي برمجك وعملك", "مين طورك")
      and not un.other_subject("ايه هي عاصمه فرنسا يا صاحبي", "عاصمه فرنسا ايه"))
r = seshat.reply("كم عدد سكان الصين")
check("مبيردش برد سؤال شبهه عن حاجة تانية",
      r["fallback"] and r["kind"] == "fact" and r["searchable"] and "مصر" not in r["response"], str(r))
r = seshat.reply("مين اخترع المصعد")   # سؤال مش موجود في أي ملف معرفة (الراديو اتضاف في جولة 15)
check("سؤال معلومة مش عنده: مبيألفش إجابة شريفة",
      r["fallback"] and "أوتيس" not in r["response"]
      and any(w in r["response"] for w in ("مين اخترع المصعد", "مش متأكد", "مش هألف", "تقصد")),
      str(r))

print("\n9) الرسم النصي (data/drawings.py)")

rows = drawings.grid_for("heart", "R")
check("شبكة الرسمة 16 × 16", len(rows) == 16 and all(len(x) == 16 for x in rows))
text = drawings.answer("cat", "K")
check("الرسمة المكتوبة بتتفك لنفس الشبكة", drawings.parse_grid(text) == drawings.grid_for("cat", "K"))
check("الرسمة المكتوبة أقصر من الشبكة (أقل من 60% من حروفها)",
      len(drawings.encode(drawings.design("cat"), "K")) < 0.6 * 16 * 17)
check("كل الرسومات بتتكتب وتتفك صح", all(
    drawings.parse_grid(drawings.answer(k, c)) == drawings.grid_for(k, c)
    for k, c in drawings.combos() + drawings.combos(True)))
check("اللون الأساسي بيتغير من أول سطر بس",
      drawings.encode(drawings.design("car"), "R").split("\n")[1:]
      == drawings.encode(drawings.design("car"), "B").split("\n")[1:])
check("عين القطة السودا بتبان (لون مختلف عن الجسم)", "W" in "".join(drawings.grid_for("cat", "K")))
check("رسمة بايظة بتترفض", drawings.decode("```draw\n#=R\na3R\n```") is None
      and drawings.decode("```draw\na16.\n```") is None)
skipped = text.replace("\nc=", "", 1)
check("سطر متكتبش: الباقي بيفضل في مكانه", drawings.parse_grid(skipped) is None
      and drawings.parse_grid_loose(skipped)[0][3:] == drawings.grid_for("cat", "K")[3:])
ragged = text.replace("5.6#5.", "5.6#4.", 1)
check("القراية الصارمة بترفض سطر ناقص والمتسامحة بتكمله",
      drawings.parse_grid(ragged) is None and drawings.parse_grid_loose(ragged)[1] == 1)
check("فهم الطلب: رسمة ولون", drawings.what_was_asked("ارسم قلب احمر") == ("heart", "R"))
check("فهم الطلب: كذا رسمة", drawings.find_objects("ارسم قطه سودا وشمس وعربيه زرقا")
      == [("cat", "K"), ("sun", None), ("car", "B")])
check("فهم الطلب: إنجليزي", drawings.find_objects("draw a black cat and a red car")
      == [("cat", "K"), ("car", "R")])
check("فهم الطلب: حاجة مش معروفة", drawings.find_objects("ارسم ديناصور") == [])
check("مفيش تركيبة اختبار في التدريب", not set(drawings.combos()) & set(drawings.combos(True)))
two = drawings.read_full(drawings.full_block([drawings.grid_for("cat"), drawings.grid_for("sun")]))
check("رسمتين جنب بعض", len(two) == 16 and all(len(x) == 33 for x in two))
r = seshat.draw("ارسم ديناصور")
check("خ4: حاجة ماتعلمهاش → أقرب رسمة بيتعرضها من غير رفض",
      not r["fallback"] and "أقرب" in r["response"] and "ديناصور" in r["response"], str(r))
check("خ4: الديناصور بيتفسر لأقرب رسمة (قطه)",
      drawings.nearest_objects("ارسم ديناصور")[0][0][0] == "cat", str(r))

print("\n10) السياق الموسّع: الرسايل الطويلة والسؤال جواها (engine.extract_question)")
from engine import extract_question

# الاستخراج نفسه
check("رسالة قصيرة بتترجع زي ما هي", extract_question("اسمك ايه") == "اسمك ايه")
check("تحية + سؤال → السؤال", extract_question("مساء الخير يا سشات ازيك عامل ايه عاصمة اليابان ايه").endswith("عاصمة اليابان ايه"))
check("سؤال + كلام بعده → السؤال في الأول", extract_question("عاصمة مصر ايه وبالمناسبة انا تعبان النهاردة كتير من الشغل").startswith("عاصمة مصر ايه"))
check("رسالة سطرين والسؤال في آخر سطر", extract_question("مساء الخير\nان جدعنة دلوقتي الوقت متاخر\nمين بنى الهرم الاكبر").endswith("مين بنى الهرم الاكبر"))
check("رسالة إنجليزي طويلة", "capital of japan" in extract_question(
    "hey seshat i hope you are fine today " * 8 + "what is the capital of japan").lower())

# الرد على رسايل طويلة (الذاكرة بتلاقي السؤال المستخرج)
LONG = ("مساء الخير يا سشات ازيك عامل ايه اتمنى تكون بخير انا دلوقتي قاعد في البيت لوحدي "
        "والجو بردان شوية وقعدت افكر في سؤال لحد ما لقيت انك انت الوحيد اللي ممكن يساعدني "
        "فيه سمعت عنك كتير وصاحبي قال لي انك بتجاوب صح دايمًا المهم خلاص كفاية كلام فاضي ")
r = seshat.reply(LONG + "عاصمة مصر ايه")
check("رسالة 300+ حرف: عاصمة مصر", not r["fallback"] and "القاهرة" in r["response"], r["response"][:50])
r = seshat.reply(LONG * 3 + "عاصمة اليابان ايه")
check("رسالة 900+ حرف: عاصمة اليابان", not r["fallback"] and "طوكيو" in r["response"], r["response"][:50])
r = seshat.reply(LONG + "154 + 382")
check("رسالة طويلة فيها حساب", not r["fallback"] and "536" in r["response"], r["response"][:50])
check("السياق في الملف 4096", seshat.config["block_size"] == 4096, str(seshat.config["block_size"]))

print("\n11) إصلاحات بحث ويكي (جولة 15): قايمة استبعاد كلمات الاستفهام")
import brain as _brain
check("«مين اخترع الراديو» → البحث بالموضوع بس",
      _brain._wiki_search_query("مين اخترع الراديو") == "الراديو",
      _brain._wiki_search_query("مين اخترع الراديو"))
check("«مين اكتشف بنسلين» → «بنسلين»",
      _brain._wiki_search_query("مين اكتشف بنسلين") == "بنسلين")
check("كلمات المقارنة بتتشال من بحث الفرق",
      "الفرق" not in _brain._wiki_search_query("ايه الفرق بين الفيزياء الكمية والفيزياء النيوتنية")
      and "الكمية" in _brain._wiki_search_query("ايه الفرق بين الفيزياء الكمية والفيزياء النيوتنية"))
check("صفحة الراديو بتعدى تقييم العلاقة بعد الإصلاح",
      _brain._wiki_relevant("مين اخترع الراديو", "الراديو",
                            "الراديو هو جهاز بيبث الصوت عن طريق موجات كهرومغناطيسية.") >= 0.6)
check("صفحة ملهاش علاقة لسه بترفض زي الأول",
      _brain._wiki_relevant("مين اخترع الراديو", "الكونكورد",
                            "طائرة الكونكورد نفاثة نقل مدني أسرع من الصوت.") < 0.6)
check("مهلة DDG أقصر من مهلة ويكي (مايعلقش الرد)",
      min(8, _brain._NET_TIMEOUT) == 8 and _brain._NET_TIMEOUT >= 8)

print("\n12) جولة خ11: مولّد الأكواد الضخمة + تكوين الجمل + سياق المحادثة")

# --- مولّد الأكواد الضخمة: قوالب بتتكوّن لحظيًا، مضمونة من غير نموذج ولا نت ---
r = b.reply("اكتب كود لعبة اكس او")
check("خ11 كود: «اكتب كود لعبة اكس او» = برنامج كامل من المولد",
      r["source"] == "big_code" and "class TicTacToe" in r["response"]
      and "def " in r["response"], str(r)[:120])
check("خ11 كود: حجم الكود فعلاً ضخم (أكتر من 60 سطر وأكتر من 2500 حرف)",
      r["response"].count("\n") > 60 and len(r["response"]) > 2500,
      f"{r['response'].count(chr(10))} سطر — {len(r['response'])} حرف")

r = b.reply("اكتب كود تطبيق مهام بفلاسك")
check("خ11 كود: تطبيق مهام flask فيه مسارات REST",
      r["source"] == "big_code" and "Flask" in r["response"]
      and "app.route(" in r["response"].lower(), str(r)[:120])

r = b.reply("اكتب كود نظام إدارة مكتبة")
check("خ11 كود: نظام مكتبة بكلاسات ودوال كتير",
      r["source"] == "big_code" and "class Library" in r["response"]
      and r["response"].count("def ") >= 5, str(r)[:120])

r = b.reply("اكتب كود خوارزميات الترتيب")
check("خ11 كود: خوارزميات الترتيب والبحث",
      r["source"] == "big_code" and "merge_sort" in r["response"]
      and "binary_search" in r["response"], str(r)[:120])

r = b.reply("اكتب كود آلة حاسبة")
check("خ11 كود: آلة حاسبة بمحلل تعبيرات",
      r["source"] == "big_code" and "class Calculator" in r["response"], str(r)[:120])

r = b.reply("make me a python calculator")
check("خ11 كود: طلب إنجليزي بيتفهم برضه",
      r["source"] == "big_code" and "class Calculator" in r["response"], str(r)[:120])

r = b.reply("اكتب كود مركب فضاء")
check("خ11 كود: موضوع بره القوالب = دليل صريح بالقوايم (مش أكواد مخترعة)",
      r["source"] == "big_code" and "تطبيق مهام" in r["response"]
      and "لعبة إكس-أو" in r["response"], str(r)[:120])

r = b.reply("يعني ايه كود نظيف")
check("خ11 كود: كلام عام عن الأكواد ما بيدخلش المولد",
      r["source"] != "big_code", r["source"])

# --- تكوين الجمل: تركيبات جديدة بنفسه من بنوك الكلمات ---
r = b.reply("كوّن جملة")
check("خ11 جمل: «كوّن جملة» = تركيبة جديدة + شرح التركيب",
      r["source"] == "compose" and "1)" in r["response"]
      and ("فاعل" in r["response"]), str(r)[:120])

r = b.reply("اكتب جملة عن البحر")
check("خ11 جمل: الموضوع بيدخل جوه التركيبة",
      r["source"] == "compose" and "البحر" in r["response"]
      and "2)" in r["response"] and "3)" in r["response"], str(r)[:120])

sent1 = b.reply("كوّن جملة")["response"]
sent2 = b.reply("اكتب جملة عن الفضاء")["response"]
check("خ11 جمل: كل نداء بيجيب تركيبة جديدة (مش منسوخة)",
      sent1 != sent2 and "الفضاء" in sent2, sent2[:80])

r = b.reply("compose a sentence about the sea")
check("خ11 جمل: إنجليزي — subject + verb + object",
      r["source"] == "compose" and "sea" in r["response"]
      and "subject" in r["response"], str(r)[:120])

# --- سياق المحادثة: المتابعات القصيرة بترجع لآخر موضوع ---
b_ctx = Brain(_DummyEngine())
b_ctx.learned = LearnedMemory(os.path.join(tmp_dir, "learned_ctx.json"))
b_ctx.cfg.api_key = ""
b_ctx.cfg.search_enabled = False
b_ctx.reply("كلمني عن القاهرة")
r = b_ctx.reply("كمان")
check("خ11 سياق: «كمان» بعد موضوع بيرجع للموضوع نفسه",
      r["source"] == "context" and "القاهرة" in r["response"], str(r)[:120])
r = b_ctx.reply("طب")
check("خ11 سياق: «طب» بيفضل على نفس الموضوع",
      r["source"] == "context" and "القاهرة" in r["response"], str(r)[:120])
check("خ11 سياق: سؤال جديد بيبدأ موضوع جديد",
      _brain._topic_of("كلمني عن الاسكندرية") == "الاسكندرية"
      and _brain._topic_of("كمان") == "", str(_brain._topic_of("كمان")))

# --- سياق الإدخال والإخراج: رسايل طويلة جدًا وردود ضخمة ---
big_msg = "مساء الخير يا سشات ازيك عامل ايه اتمنى تكون بخير انا دلوقتي قاعد في البيت " * 170
r = b.reply(big_msg)                       # رسالة ~8000 حرف
check("خ11 سياق: رسالة ~8000 حرف بتتقبل وبتترود",
      isinstance(r, dict) and "response" in r, str(r)[:80])
check("خ11 سياق: حد الإدخال في الواجهة 8000 حرف",
      "SESHAT_MAX_MESSAGE_CHARS" in open("app.py", encoding="utf-8").read())

# --- حمايات خ11: النموذج نصي + مفيش بقايا خوفو في البيانات الحية ---
def _live_data_text():
    blob = ""
    for f in ("data/ar_chat.txt", "data/ar_knowledge.txt", "data/en_chat.txt"):
        blob += open(f, encoding="utf-8").read().lower()
    return blob

live = _live_data_text()
check("خ11 حماية: مفيش ادعاء خوفو/توليد صور في بيانات الحوار الحية",
      "خوفو" not in live and "khufu" not in live
      and "diffusion" not in live and "750px" not in live)
check("خ11 حماية: ملفات المولدات الجديدة موجودة وبيان الحالة صحيح",
      os.path.exists("data/big_code.py") and os.path.exists("data/sent_composer.py")
      and b.status()["big_code"] is True and b.status()["compose"] is True
      and b.status()["images"] is False)

# ============================================================
# خ12 — السياق الممتد: تدفق ملايين الحروف بذاكرة ثابتة
# ============================================================
print("\n15) خ12 — التدفق الطويل (نافذة منزلقة + مراسي انتباه)")
from engine import NumpySeshat

info = seshat.info()
check("خ12: السياق الأقصى في المعلومات = ٢٠٠٠٠٠٠",
      info.get("max_context") == 2_000_000, str(info.get("max_context")))
check("خ12: reply_long موجود في المحرك", hasattr(seshat, "reply_long"))

# مولّد المستندات: كل سؤال عن معلومة في آخر المستند
sys.path.insert(0, "data")
import long_docs

_ld = long_docs.build(seed=20260, count=40)
check("خ12: مولّد المستندات بيطلع أمثلة سليمة",
      all("\n" in e["question"] and e["answer"] and e["think"].startswith("مستند |")
          for e in _ld) and len(_ld) == 40)
_over_win = sum(1 for e in _ld if len(e["question"]) > info.get("stream_window", 768) + 40)
check("خ12: أغلب المستندات أطول من النافذة", _over_win >= 25, f"{_over_win}/40")

# ذاكرة التدفق ثابتة الحجم مهما طال النص
s_np = seshat if isinstance(seshat, NumpySeshat) else NumpySeshat()
sc = s_np._stream_cache()
_stream_ids = [3 + (i % 150) for i in range(6000)]        # ٦٠٠٠ حرف — ٨× النافذة
for _s in range(0, len(_stream_ids), 512):
    s_np._forward_stream(_stream_ids[_s:_s + 512], sc)
check("خ12: ذاكرة التدفق ثابتة بعد ٦٠٠٠ حرف",
      len(sc["order"]) == sc["capacity"] and sc["count"] == 6000,
      f"{len(sc['order'])}/{sc['capacity']} count={sc['count']}")
check("خ12: المقاعد كلها موجودة مرة واحدة بالظبط",
      sorted(sc["order"]) == list(range(sc["capacity"])))
check("خ12: المراسي في أول الترتيب دايمًا",
      sc["order"][:sc["n_sink"]] == list(range(sc["n_sink"])))

# reply_long: مستند + سؤال عن آخره (من غير تدريب على longdoc لسه — المهم
# إن المسار يشتغل من أول التدفق للرد ومن غير ما الرام تتملي)
_doc = "تقرير عن النظام:\n" + ("سطر بيانات عادي مش مهم.\n" * 300)
r = s_np.reply_long(_doc, "ايه موضوع التقرير", use_fallback=False)
check("خ12: reply_long بيرجع رد كامل الشكل",
      isinstance(r, dict) and "response" in r and "thinking" in r
      and "confidence" in r and "doc_chars" in r and r["doc_chars"] > 4000, str(r)[:120])
check("خ12: التدفق ما كاسرش block_size في التدفق الطويل", True)

# brain: الرسالة الضخمة بتروح لمسار السياق الممتد (مسموح يبقى فيها fallback)
_big = ("رسالة طويلة جدًا من العميل:\n" + "كلام كتير مش مهم هنا.\n" * 400
        + "الخلاصة: عايز أعرف التقرير جاهز امتى")
rb = b.reply(_big)
check("خ12: رسالة ضخمة في المخ بتتقبل وبتترود",
      isinstance(rb, dict) and "response" in rb, str(rb)[:120])

# توليد طويل: نفس المسار بتوليد من غير حد block_size
_gen = s_np._generate_long(s_np.tok.encode_prompt("اسمك ايه"), 40, 0.0)
check("خ12: التوليد الطويل شغال (٤٠ حرف من غير قفل block)",
      isinstance(_gen[0], list) and len(_gen[0]) <= 40, str(len(_gen[0])))

# حد الـ ٢م في الواجهة
check("خ12: سيرفر فيه /api/long وحد السياق ٢م",
      "api/long" in open("app.py", encoding="utf-8").read()
      and "16 * 1024 * 1024" in open("app.py", encoding="utf-8").read())

# ============================================================
# خ15 — الأدوات النصية المتقدمة المحلية (text_tools)
# ============================================================
print("\n16) خ15 — الأدوات المتقدمة: وحدات/أرقام بالحروف/إحصاء/تلخيص/تواريخ/كلمات مرور")
from text_tools import (try_text_tool, convert_units, number_to_arabic_words,
                        summarize, text_stats, num_stats, gen_password,
                        password_strength, date_diff, digits_convert, day_of_year)

check("خ15: طول كم→م", convert_units("حوّل 5 كيلومتر الى متر") and "5000" in convert_units("حوّل 5 كيلومتر الى متر"))
check("خ15: حرارة F→C", "37.7" in convert_units("حوّل 100 فهرنهايت الى سيلسيوس"))
check("خ15: بيانات GB→MB", "3072" in convert_units("حوّل 3 جيجابايت الى ميجابايت"))
check("خ15: سرعة كم/س→م/ث", "25" in convert_units("حوّل 90 كم/ساعة الى متر/ثانية"))
check("خ15: متر إلى كم بالعكس", convert_units("حوّل 7 متر الى كم") and "0.007" in convert_units("حوّل 7 متر الى كم"))
check("خ15: رقم بالحروف مركب", number_to_arabic_words(125) == "مائة وخمسة وعشرون")
check("خ15: آلاف الجمع", "آلاف" in number_to_arabic_words(3005))
check("خ15: صفر بالحروف", number_to_arabic_words(0) == "صفر")
check("خ15: إحصاء numpy", num_stats([4, 8, 15, 16, 23, 42]) and "المتوسط: 18" in num_stats([4, 8, 15, 16, 23, 42]))
_s15 = ("مصر بلد فيه تاريخ كبير. النيل بيمر في وسطها وسكانها كتير جدًا. "
        "القاهرة عاصمتها وأكبر مدنها على الإطلاق. الزراعة بتعتمد على النيل من آلاف السنين. "
        "السياحة من أهم مصادر الدخل القومي والاقتصاد بيحاول ينمو بسرعة.")
check("خ15: تلخيص استخراجي", summarize(_s15) and len(summarize(_s15).splitlines()) <= 4)
check("خ15: تحليل نص", "عدد الكلمات" in text_stats("مرحبا بكم في النص التجريبي السريع"))
check("خ15: فرق تواريخ", "59 يوم" in date_diff((2026, 1, 1), (2026, 3, 1)))
check("خ15: رقم اليوم في السنة", "اليوم رقم" in day_of_year((2026, 11, 19)))
_pw15 = gen_password(20)
check("خ15: توليد كلمة مرور", len(_pw15) == 20 and any(c.isdigit() for c in _pw15) and any(c.isupper() for c in _pw15))
check("خ15: قياس قوة المرور", "قوية" in password_strength("Abc123!@#xY"))
check("خ15: أرقام هندية→إنجليزية", "65" in digits_convert("سعر ٦٥ جنيه"))
check("خ15: الموزع العام للأدوات", try_text_tool("حوّل 5 كيلومتر الى متر") and "5000" in try_text_tool("حوّل 5 كيلومتر الى متر")[0])

# خ17 — الفهم وتكوين الجمل: الصيغ المفتوحة الطبيعية + ارتباط الموضوع + الرأي المؤلف (طلب المستخدم الصريح)
_c17 = b.reply("كوّن جملة عن البحر")
_c17txt = _c17.get("response", "")
check("خ17: تكوين الجمل بيغطي الموضوع في التلات جمل", _c17txt.count("بحر") >= 3, f"عدد الموضوع={_c17txt.count('بحر')}")
check("خ17: مكان الموضوع الطبيعي (البحر → الشاطئ)", "الشاطئ" in _c17txt, _c17txt[:60])
_c17b = b.reply("اتكلم في سطرين عن الكتابة")
check("خ17: «اتكلم في سطرين عن X» بيولد جمل مواضيعها نظيفة", "الكتابة" in _c17b.get("response", "")
      and "سطرين عن" not in _c17b.get("response", ""), _c17b.get("response", "")[:70])
_c17c = b.reply("اكتب لي فكرة جديدة عن الصداقة")
check("خ17: «اكتب لي فكرة جديدة عن X» مش بيرفض", "مش قادر أحدد" not in _c17c.get("response", "")
      and "الصداقة" in _c17c.get("response", ""), _c17c.get("response", "")[:70])
_c17d = b.reply("ايه رأيك في القراءة")
check("خ17: «ايه رأيك في X» بيولد رأي مؤلف معلن", "رأيي في" in _c17d.get("response", "")
      and "مش قادر أحدد" not in _c17d.get("response", "")
      and "تكويني الشخصي" in _c17d.get("response", ""), _c17d.get("response", "")[:80])
_c17e = b.reply("اتكلم عن النيل")
check("خ17: «اتكلم عن X» بيربط الموضوع بمكانه الطبيعي", "النيل" in _c17e.get("response", "")
      and "قرب النهر" in _c17e.get("response", ""), _c17e.get("response", "")[:70])
_rb15 = b.reply("حوّل 3 جيجابايت الى ميجابايت")
check("خ15: المخ بيوجه للأداة المتقدمة", _rb15.get("source") == "tools" and "3072" in _rb15.get("response", ""),
      str(_rb15.get("source")))

print("\n" + "—" * 40)
if FAILURES:
    print(f"❌ {len(FAILURES)} اختبار فشل: {', '.join(FAILURES)}")
    sys.exit(1)
print("✅ كل الاختبارات عدت")
