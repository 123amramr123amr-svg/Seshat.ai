# -*- coding: utf-8 -*-
"""سكربت جولة خ8 — مدخلات معرفة عن النموذج الموحّد seshat 1.0 + خوفو + قوس قزح.

- بيتأكد إن مفيش سؤال مكرر (فحص بالسطر الأول لكل مدخل).
- غلافين طويلين (450–1050 حرف) واحد عربي وواحد إنجليزي.
- بينضف learned.json من أي كاش ويب قديم (بيسيب user/llm بس).
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def load_pairs(path, q_prefix):
    qs = []
    if not os.path.exists(path):
        return qs
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith(q_prefix):
                qs.append(s[len(q_prefix):].strip())
    return qs


def append_group(path, pairs, q_prefix):
    """بيضيف مجموعة جديدة آخر الملف: فاصل --- ثم أزواج س/ج أو q/a."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    existing = [q.lower() for q in load_pairs(path, q_prefix)]
    out_lines = []
    added = 0
    for q, a in pairs:
        key = q.lower()
        if any(key in e or e in key for e in existing):
            print(f"   ⚠️  تخطي مكرر: {q[:40]}…")
            continue
        out_lines.append("")
        out_lines.append(f"{q_prefix}: {q}")
        out_lines.append(f"{'ج' if q_prefix == 'س' else 'a'}: {a}")
        added += 1
        existing.append(key)
    if not added:
        print(f"   مفيش جديد في {os.path.basename(path)}")
        return 0
    with open(path, "a", encoding="utf-8") as f:
        if not content.endswith("\n"):
            f.write("\n")
        f.write("---\n")
        f.write("\n".join(out_lines).lstrip("\n"))
        f.write("\n")
    print(f"   ✅ {os.path.basename(path)}: +{added} مدخل")
    return added


AR_KNOWLEDGE = [
    ("ما هو النموذج الموحد seshat 1.0؟",
     "seshat 1.0 هو النموذج الموحد بتاع المستودع: عقل سي شات النصي (حوار، حساب، تعلم، بحث، أدب) "
     "ومولد الصور خوفو شغالين في مدخل واحد. لما تكتب سؤال بيتم فهمه ويرد من الذاكرة أو النموذج، "
     "ولما تكتب «ارسم…» نفس التحليل بيوجه الطلب لخوفو اللي بيرسم على 750×750. كله محلي على نفس "
     "السيرفر من غير أي خدمة خارجية، ومن نفس الذاكرة والمفاتيح المشتركة."),
    ("منين بتيجي الصور في seshat 1.0؟",
     "كل الصور بتيجي من خوفو نفسه — نموذج انتشار (DDPM) ورسام إجرائي هجين مكتوبين من الصفر في "
     "المستودع. مفيش مواقع صور خارجية ولا مفاتيح ولا انتظار شبكة، والصورة بتتخزن في data/generated "
     "وتتعرض من عنوان السيرفر نفسه. نفس البذرة ونفس الوصف = نفس الصورة بالحرف، فالتعديل بالوصف "
     "مضبوط ومتوقع."),
    ("إزاي يفهم seshat 1.0 طلب الصورة بالظبط؟",
     "الطلب بيمر على خمس مراحل بحث: تطابق تام مع الأصناف، ثم مرادفات عربي/إنجليزي داخل النص، "
     "ثم أقرب كلمة أبجديًا، ثم دلوات دلالية (زي «قرد» → طيور)، ثم أقرب صنف كاحتياطي — خوفو عمره "
     "ما يقول «مش عارف أرسم». ولو الوصف فيه بيئة وجو وعناصر («قارب في البحر وقت الغروب») بيتحلل "
     "لتركيبة واحدة: خلفية بحر + تلطيخ غروب + قارب، بأعداد من الأرقام أو الكلمات أو الجمع."),
    ("ليه صور خوفو بقت أوضح وألوانها أعلى؟",
     "في جولة خ8 اتعدل الخلط الهجين: الرسمة الإجرائية الحادة بقت 82% من الصورة والأجواء المتعلمة "
     "من النموذج 18%، والترويس بقى أقوى، وفوقهم طبقة طاقة بتزود التشبع اللوني 1.20 مرة ومنحنى "
     "تباين ناعم (smoothstep) بيقوي الفرق بين الظل والنور. النتيجة: شكل مقروء واضح وألوان أعلى "
     "بطاقة من غير ما التفاصيل تتحرق."),
    ("ما هو قوس قزح وليه ألوانه سبعة؟",
     "قوس قزح ظاهرة بصرية بتحصل لما الضوء بينكسر وانعكس جوه قطار المطر، وبيتفرق لألوان الطيف "
     "حسب طول الموجة: أحمر، برتقالي، أصفر، أخضر، أزرق، نيلي، بنفسجي. الرقم سبب مش سحر — البشر "
     "بدأوا يعدوها سبعة من أيام نيوتن، والحقيقة إن الطيف متصل ومفيش حدود حادة بين الألوان. القوس "
     "بينشوف لما الشمس ورانا والرش قدامنا، وبيبان أكتر وقت الصبح والعصر."),
    ("أقدر أعلّم seshat 1.0 معلومات جديدة؟",
     "أيوه — بصيغة «علمني: س: السؤال ج: الجواب» بيتخزن في data/learned.json على طول، والمرة اللي "
     "جاية بيرد من الذاكرة بأولوية على كل حاجة. و«انسى: السؤال» بيحذفه. التعليم بيشتغل على نفس "
     "النموذج الموحد: سؤالك النصي وصورك كلها من نفس الذاكرة والهوية."),
    ("ليه seshat 1.0 مش بيستخدم مواقع صور خارجية؟",
     "لثلاث حاجات: الخصوصية (الوصف بتاعك مش بيخرج بره السيرفر)، والسرعة (مفيش انتظار 45 ثانية "
     "ولا حدود يومية)، والملكية (الرسام والنموذج مكتوبين في المستودع وممكن تتطور مع كل جولة تدريب "
     "زي ما حصل من خ2 لخ8). أي صورة بتتولد من دماغ خوفو اللي اتدرب على مشاهد إجرائية."),
]

AR_KNOWLEDGE.append(
    ("احكيلي بالتفصيل رحلة طلب الصورة من أول حرف لحد الصورة النهائية في النموذج الموحد",
     "أول ما تكتب «ارسم قارب في البحر وقت الغروب» الرسالة بتدخل خط الفهم الموحد: الأدب الأول "
     "(مفيش شتم هنا)، الحساب مش متفعل، محلل الصور بيطلع الوصف. بعدين خوفو بيبدأ بحثه خمس مراحل "
     "على النص المتطبّع: تطابق تام، مرادفات («البحر» بيلاقي sea)، أقرب كلمة، دلوات دلالية، واحتياطي "
     "— فيطلع مواصفة تركيبة: بيئة بحر + جو غروب + قارب واحد. الرسام الإجرائي بيرسم البحر والأفق "
     "والقارب بأشكال حادة على دقة المخرج كاملة (750×750)، وفي نفس الوقت شبكة الانتشار بتفسّر «جو» "
     "المشهد من أوزانها هي — ألوان اتعلمتها وهي بتتدرب مش ألوان متكتبة. الاتنين بيتخلطوا 82% رسمة "
     "حادة + 18% جو، وبعدين ترويس حواف وطبقة طاقة بتزود التشبع والتباين. الصورة بتتحفظ PNG مع "
     "الوصف والبذرة، والرد بيرجع فورًا بلينك محلي — من غير نت ولا انتظار.")
)

AR_CHAT = [
    ("انت مين بالظبط؟",
     "أنا seshat 1.0 — النموذج الموحد: عقل سي شات للفهم والحوار والحساب والتعلم، ومولد الصور خوفو "
     "شغال جوايا. مطوريني عمرو أحمد، وكل حاجة فيا محلية."),
    ("انت بتصور إزاي؟ | الصور بتطلع منين؟",
     "برسم بنفسي من خوفو — نموذج صور محلي اتعلم مشاهد إجرائية. مفيش مواقع خارجية خالص: كل صورة "
     "بتتولد على السيرفر نفسه وتتخزن عندنا."),
    ("ارسم قوس قزح فوق البحر | صورة قوس قزح",
     "اتفضل — قوس قزح بألوانه السبعة فوق المية، هيرسمها خوفو بجوهرة إجرائية حادة وجو متعلم. جرب "
     "كمان «قوس قزح فوق الجبال» أو «طيور فوق الجبال مع قوس قزح»."),
    ("صورك بقت أحلى ليه؟ | ليه الصور واضحة؟",
     "جولة خ8 رفعت وزن الرسمة الحادة لـ 82% وزودت طبقة طاقة: تشبع أعلى وتباين ناعم. الشكل بقى "
     "مقروء وألوان أعلى."),
    ("لو طلبت حاجة معملتهاش؟",
     "هلاقي أقرب حاجة أرسمها — خوفو عمره ما يقول «مش عارف أرسم». أي وصف بيتفسر لصنف أو تركيبة "
     "مناسبة."),
    ("مين اللي عملك؟ | مطورك مين؟",
     "عمرو أحمد (Amr Ahmed) — والنموذج الموحد seshat 1.0 بيتطور جولة ورا جولة في نفس المستودع."),
]

EN_KNOWLEDGE = [
    ("what is seshat 1.0 (the unified model)?",
     "seshat 1.0 merges the text brain (chat, math, learning, search, politeness) with khufu the "
     "image generator into one local model with a single reply pipeline. text questions are "
     "answered from memory and the trained engine, and any 'draw …' request is parsed and painted "
     "by khufu at 750x750. everything runs on the same server with no external services."),
    ("where do the images come from?",
     "every image is generated locally by khufu — a small DDPM diffusion model plus a procedural "
     "hybrid renderer written from scratch in this repository. there is no external image service, "
     "no api keys, and no network wait. images are stored as png in data/generated and served from "
     "the server's own url."),
    ("how does the image request get understood?",
     "the description goes through five lookup stages: exact class match, arabic/english synonyms "
     "inside the text, closest word, semantic buckets, and a final nearest-class fallback — so "
     "khufu never refuses to draw. free-form descriptions become composites: environment + mood "
     "lighting + objects with counts from digits, arabic number words, or plurals."),
    ("why are the images sharper and more colorful now?",
     "round kh8 raised the crisp procedural weight to 82%, strengthened edge sharpening, and added "
     "a vivid layer that boosts color saturation by 1.20x and applies a smooth contrast curve. the "
     "result reads as a clear subject with brighter colors instead of a color cloud."),
]
EN_KNOWLEDGE.append(
    ("explain in detail how a drawing request travels through the unified model until the final picture",
     "first the message enters the unified intent pipeline: politeness guard, math guard, then the "
     "image parser extracts the description. khufu normalizes arabic text, then runs five lookup "
     "stages to map words to scene classes and counts. the procedural renderer draws the sky, "
     "ground and objects as sharp shapes at the full output resolution of 750x750, while the "
     "trained diffusion network produces its own learned atmosphere colors from pure noise in "
     "fifty denoising steps with clipped denoising for stability. the two layers are blended "
     "eighty-two percent crisp and eighteen percent atmosphere, then unsharp-masked and passed "
     "through the vivid saturation and contrast layer. the png is saved with the prompt and seed "
     "so the same request always reproduces the same picture, and the chat reply returns instantly "
     "with a local url — completely offline.")
)

EN_CHAT = [
    ("who are you?",
     "i am seshat 1.0 — the unified model: the seshat text brain plus the khufu image generator, "
     "developed by amr ahmed, running fully locally."),
    ("how do you make images?",
     "i draw them myself with khufu, my local diffusion and procedural renderer. no external "
     "services — every picture is generated on the same server."),
    ("draw a rainbow over the sea",
     "done — khufu paints a seven-color rainbow arc over the water with a crisp procedural scene "
     "and its own learned atmosphere. try 'rainbow over the mountains' too."),
    ("what if i ask for something you never saw?",
     "i map it to the closest known scene class — khufu never says 'i cannot draw'. every "
     "description resolves to a class or a composite."),
    ("why are your images better now?",
     "round kh8: 82% crisp procedural layer, stronger sharpening, and a vivid layer with higher "
     "saturation and smooth contrast."),
]


def clean_learned():
    path = os.path.join(DATA, "learned.json")
    if not os.path.exists(path):
        print("   learned.json مش موجود — تخطي التنظيف")
        return 0
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    items = raw if isinstance(raw, list) else list(raw.values())
    kept, removed = [], 0
    for it in items:
        if not isinstance(it, dict):
            removed += 1
            continue
        src = it.get("src", "")
        if src == "web":
            removed += 1            # كاش ويب قديم — بيتنضف كل جولة
            continue
        kept.append(it)
    if removed:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    print(f"   ✅ learned.json: اتحفظ {len(kept)} واتنضف {removed}")
    return removed


def main():
    print("خ8 — إضافة مدخلات النموذج الموحّد:")
    total = 0
    total += append_group(os.path.join(DATA, "ar_knowledge.txt"), AR_KNOWLEDGE, "س")
    total += append_group(os.path.join(DATA, "ar_chat.txt"), AR_CHAT, "س")
    total += append_group(os.path.join(DATA, "en_knowledge.txt"), EN_KNOWLEDGE, "q")
    total += append_group(os.path.join(DATA, "en_chat.txt"), EN_CHAT, "q")
    # غلافين طويلين: عربي وإنجليزي — لازم 450–1050 حرف
    longs = [AR_KNOWLEDGE[-1], EN_KNOWLEDGE[-1]]
    for q, a in longs:
        ln = len(q) + len(a)
        assert 450 <= ln <= 1050, f"طول الغلاف {ln} برا المدى (450–1050): {q[:40]}"
        print(f"   ✅ غلاف طويل {ln} حرف: {q[:40]}…")
    clean_learned()
    print(f"الإجمالي: +{total} مدخل — خلاص.")


if __name__ == "__main__":
    sys.exit(main())
