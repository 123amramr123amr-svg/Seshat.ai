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
                                    "translate", "capital", "math", "draw", "unknown"})
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
check("المخ: الرسم الخارجي مقفول افتراضيًا (مفيش صورة من شركة تانية)",
      r["source"] == "draw" and not r["image"] and r["fallback"], str(r))
b.cfg.external_images = True
r = b.reply("ارسم قرد بيغني")
check("المخ: لو اتفتح، الصورة الخارجية مكتوب عليها إنها مش رسمه",
      r["image"] and r["image"].startswith("https://image.pollinations.ai/prompt/")
      and "pollinations" in r["response"] and r["source_label"] == "خدمة رسم خارجية", str(r))
b.cfg.external_images = False
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
r = seshat.reply("مين اخترع التليفون")
check("سؤال معلومة مش عنده: بيقول كده عن السؤال نفسه",
      r["fallback"] and "مين اخترع التليفون" in r["response"], str(r))

print("\n9) الرسم والصور (data/drawings.py + vision.py)")
import base64
import numpy as np
import vision

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
check("حاجة ماتعلمهاش: بيقول بيعرف يرسم إيه", r["fallback"] and "قطه" in r["response"], str(r))
px = np.zeros((40, 60, 4), dtype=np.uint8)
px[..., 3] = 255
px[:20, :, :3] = (40, 110, 230)
px[20:, :, :3] = (50, 160, 60)
img = {"w": 60, "h": 40, "data": base64.b64encode(px.tobytes()).decode()}
check("كل التعديلات بتشتغل", all(vision.apply_edit(px, op).shape[2] == 4
                                 for op in vision.EDITS if op != "enlarge"))
check("فهم أمر تعديل مركب", vision.find_edits("غبشها جدا واقلبها")[:2] == (["blur", "flip_h"], 2.0))
check("ملف PNG سليم", vision.encode_png(px)[:8] == b"\x89PNG\r\n\x1a\n")
r = vision.handle_image("خليها ابيض واسود ولفها يمين", img)
check("تعديل صورة بيرجع صورة", r["edited"].startswith("data:image/png;base64,")
      and (r["width"], r["height"]) == (40, 60))
r = vision.handle_image("ايه ده", img)
check("وصف صورة: الألوان", "أزرق" in r["response"] and "أخضر" in r["response"], r["response"])
check("كلام ملوش علاقة بالصورة بيروح للشات", vision.handle_image("اسمك ايه", img) is None)
check("سؤال عن الصورة × تعديل", vision.wants_look("ايه ده") and vision.wants_look("")
      and not vision.wants_look("خليها ابيض واسود"))
check("الصورة بتتحول PNG للمخ الخارجي",
      vision.jpeg_free_data_url(img).startswith("data:image/png;base64,iVBOR"))
from brain import BrainConfig, llm_vision
cfg0 = BrainConfig()
cfg0.api_key = ""
check("من غير مفتاح مفيش مخ خارجي للصور", llm_vision("ايه ده", "data:image/png;base64,AA", cfg0) is None)

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

print("\n" + "—" * 40)
if FAILURES:
    print(f"❌ {len(FAILURES)} اختبار فشل: {', '.join(FAILURES)}")
    sys.exit(1)
print("✅ كل الاختبارات عدت")
