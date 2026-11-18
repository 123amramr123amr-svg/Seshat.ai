"""
خ12 — مولّد المستندات الطويلة للتدريب على السياق الممتد

الفكرة: المستندات الحقيقية (ملفات كود، تقارير، سجلات، لوائح) بتبقى أطول بكتير
من سؤال واحد. النموذج لازم يتعلم يقرا مستند كامل (لحد ٣١٠٠ حرف وقت التدريب،
ولحد ٢٠٠٠٠٠٠ حرف وقت التشغيل بالنافذة المنزلقة) ويجاوب على سؤال عن تفصيلة
مكتوبة في آخر المستند.

كل نوع مستند بيسأل عن معلومة في المنطقة الأخيرة (آخر ~٥٠٠ حرف — هي اللي
النافذة المنزلقة بتحتفظ بيها وقت التشغيل)، والباقي filler بيثير بس عشان
النموذج يتعلم يهمل اللي مش له علاقة ويلاقي اللي له علاقة.

بيانات التدريب بتتولد بـ seed ثابت، والقياس (evaluate.py --long) بيولّد
مستندات جديدة بـ seed تاني عمر النموذج ما شافها.
"""

import random

# ---------- بنوك المحتوى ----------
AR_NAMES = ["احمد", "سارة", "محمد", "فاطمه", "عمر", "نور", "خالد", "ليلى",
            "يوسف", "مريم", "حسن", "دينا", "كريم", "هاله", "طارق", "ريهام"]
EN_NAMES = ["ahmed", "sara", "omar", "nour", "laila", "yousef", "mariam",
            "hassan", "dina", "karim", "hala", "tarek", "reham", "mostafa",
            "amira", "zed"]
AR_JOBS = ["مهندس", "دكتور", "معلم", "محاسب", "مبرمج", "مصمم", "محامي",
           "صيدلي", "ممرض", "صحفي", "كاتب", "مترجم"]
EN_JOBS = ["an engineer", "a doctor", "a teacher", "an accountant",
           "a programmer", "a designer", "a lawyer", "a pharmacist",
           "a nurse", "a journalist", "a writer", "a translator"]
AR_CITIES = ["القاهره", "الاسكندريه", "الجيزه", "المنصوره", "طنطا", "اسيوط",
             "المنيا", "دمياط", "بورسعيد", "السويس", "الفيوم", "زقازيق"]
EN_CITIES = ["cairo", "alexandria", "giza", "mansoura", "tant", "asyut",
             "minya", "damietta", "portsaid", "suez", "fayoum", "zagazig"]

# دوال الكود: (الاسم، الوسايط، السطر، وصف الناتج عربي، وصف الناتج إنجليزي)
CODE_FUNCS = [
    ("add", "a, b", "return a + b", "مجموع الرقمين", "the sum of the two numbers"),
    ("sub", "a, b", "return a - b", "الفرق بين الرقمين", "the difference of the two numbers"),
    ("mul", "a, b", "return a * b", "حاصل ضرب الرقمين", "the product of the two numbers"),
    ("square", "n", "return n * n", "مربع الرقم", "the square of the number"),
    ("is_even", "n", "return n % 2 == 0", "هل الرقم زوجي ولا لأ", "whether the number is even"),
    ("last_char", "s", "return s[-1]", "آخر حرف في النص", "the last character of the string"),
    ("first_char", "s", "return s[0]", "أول حرف في النص", "the first character of the string"),
    ("shout", "s", "return s.upper()", "النص كله بحروف كبيرة", "the string in upper case"),
    ("quiet", "s", "return s.lower()", "النص كله بحروف صغيرة", "the string in lower case"),
    ("length", "s", "return len(s)", "عدد حروف النص", "the length of the string"),
    ("reverse", "s", "return s[::-1]", "النص معكوس", "the string reversed"),
    ("double_list", "xs", "return [x * 2 for x in xs]", "القايمة كلها مضروبة في ٢",
     "the list with every item doubled"),
    ("total", "xs", "return sum(xs)", "مجموع عناصر القايمة", "the sum of the list items"),
    ("biggest", "xs", "return max(xs)", "أكبر عنصر في القايمة", "the largest item in the list"),
    ("smallest", "xs", "return min(xs)", "أصغر عنصر في القايمة", "the smallest item in the list"),
    ("avg", "xs", "return sum(xs) / len(xs)", "متوسط عناصر القايمة", "the average of the list items"),
]

# خ12: تعليقات حشو لملفات الكود عشان المستند يفضل أطول من نافذة التدفق (٧٦٨ حرف)
CODE_FILLERS = ["# الوحدات دي بتتستدعي من باقي الملفات في المشروع",
                "# حط تعليق واضح قبل أي تعديل هنا",
                "# دول اتعملت عشان الاختبارات تشتغل زي ما هي",
                "# ماتغيرش الترتيب — الترتيب جزء من العقد",
                "# this section is imported by other modules",
                "# keep the order — tests depend on it"]

AR_INSTR = ["اقفل الملف بعد ما تخلص", "احفظ نسخة من البيانات كل يوم",
            "راجع الأرقام قبل ما تبعته", "اكتب تاريخ اليوم في أول سطر",
            "متشاركش الحساب مع حد بره الفريق", "حدث القايمة كل اسبوع",
            "بلغ المدير لو حصل خطأ", "امسح الملفات المؤقته كل شهر",
            "استخدم كلمة السر القوية", "متفتحش لينكات مش معروفة",
            "رتب المهام بالأهميه", "خد بريك كل ساعتين",
            "اكتب ملخص الاجتماع في الملف", "جرب الكود قبل ما ترفعه",
            "خلي نسخة احتياطية قبل التحديث", "راقب الاداء كل صباح",
            "نضف الجداول القديمة", "وثق أي تغيير في الإعدادات",
            "متتأخرش عن الستاند اب اليومي", "راجع الرسايل مرتين قبل الارسال",
            "قسم المشاريع الكبيرة لمهام صغيرة", "اكتب اسم المستخدم في التقرير",
            "خلي الملفات المشتركة في الفولدر المركزي", "راجع التصميم مع الفريق قبل التنفيذ",
            "سجل الوقت اللي بتاخده في كل مهمه", "متستخدمش شبكة واي فاي عامة للشغل",
            "حدث كلمة السر كل تلت شهور", "خد نسخة من مفاتيح الـ API في مكان امن",
            "راجع اللوج قبل ما تقفل الشفت", "اتاكد ان النسخ الاحتياطي شغال",
            "متسألش لو مش فاهم — اسأل", "قفل الادوات اللي مش بتستخدمها",
            "اكتب ملاحظاتك في ملف الملاحظات", "رتب مكتبك قبل ما تمشي"]
EN_INSTR = ["close the file when you finish", "save a copy of the data every day",
            "check the numbers before sending", "write today date on the first line",
            "never share the account outside the team", "update the list every week",
            "tell the manager if an error happens", "delete temp files every month",
            "use a strong password", "do not open unknown links",
            "sort tasks by importance", "take a break every two hours",
            "write the meeting summary in the file", "test the code before you push",
            "keep a backup before updating", "watch the performance every morning",
            "clean the old tables", "document any change in settings",
            "do not be late to the daily standup", "read messages twice before sending",
            "split big projects into small tasks", "write the user name in the report",
            "keep shared files in the central folder", "review the design with the team first",
            "log the time you spend on each task", "never use public wifi for work",
            "change the password every three months", "keep api keys in a safe place",
            "check the log before closing the shift", "make sure the backup is running",
            "ask when you do not understand", "close the tools you are not using",
            "write your notes in the notes file", "tidy your desk before leaving"]

AR_EVENTS = ["المستخدم فتح الصفحة الرئيسيه", "النظام عمل نسخه احتياطيه",
             "وصلت رساله جديده من العميل", "السيرفر اتعمل له restart",
             "اتسجل دخول جديد", "اترفع ملف جديد للمكتبه",
             "التقرير اليومي اتبعت للمدير", "اتضاف مستخدم جديد للفريق",
             "الفلتر اتغير في البحث", "قاعده البيانات اتنضفت",
             "اتعطل اتصال مؤقت بالنت", "التذكرة رقم 12 اتقفلت",
             "اتعمل تحديث للنسخه", "اتسجل خطأ في الصفحه الثالثه واتصلح",
             "العميل طلب عرض سعر جديد", "اتشال جدول قديم من القاعده",
             "اتضاف تقرير اسبوعي جديد", "المخزن اتحدث بعد الشحنه الجديدة",
             "اتقفل حساب قديم بعد الطلب", "اترفع فيديو للصفحه التعليميه",
             "اتضاف سؤال جديد للأسئلة الشايعة", "الاستقبال اتصل بجهاز الشبكة واتصلح",
             "اتسجل تسليم جديد للمشروع", "الرسايل المعلقة اتبعتت كلها",
             "اتعمل مراجعة للحسابات", "اتضاف تعليق جديد على المقال"]
EN_EVENTS = ["user opened the home page", "system made a backup",
             "a new message arrived from the client", "server restarted",
             "a new login was recorded", "a new file was uploaded",
             "the daily report was sent", "a new member joined the team",
             "the search filter changed", "the database was cleaned",
             "a short network drop happened", "ticket 12 was closed",
             "a new version was deployed", "an error on page three was fixed",
             "the client asked for a new quote", "an old table was removed",
             "a new weekly report was added", "the store was updated after the shipment",
             "an old account was closed on request", "a new video was uploaded",
             "a new question was added to the faq", "support fixed a network device",
             "a new delivery was recorded", "all pending messages were sent",
             "the accounts were reviewed", "a new comment was added to the post"]


def _fill(rng, pool, used):
    """بيرجع عنصر مش متكرر من البنك"""
    for _ in range(50):
        x = rng.choice(pool)
        if x not in used:
            used.add(x)
            return x
    return rng.choice(pool)


def _pad_to(doc_lines, target, rng, line_pool):
    """بيرجّع طول المستند بالحروف لحوله المستهدف بسطور filler"""
    while sum(len(l) + 1 for l in doc_lines) < target:
        doc_lines.insert(rng.randrange(len(doc_lines)), rng.choice(line_pool))
    return doc_lines


# ============================================================
# ١) بطاقة فريق: أشخاص وخصايصهم
# ============================================================
def make_roster(rng):
    ar = rng.random() < 0.5
    names = AR_NAMES if ar else EN_NAMES
    jobs = AR_JOBS if ar else EN_JOBS
    cities = AR_CITIES if ar else EN_CITIES
    used, people = set(), []
    for _ in range(rng.randint(9, 14)):
        p = {"name": _fill(rng, names, used),
             "job": rng.choice(jobs), "city": rng.choice(cities),
             "age": rng.randint(23, 55), "num": rng.randint(1, 99)}
        if ar:
            people.append(f"- {p['name']}: شغله {p['job']}، ساكن في {p['city']}، "
                          f"عنده {p['age']} سنة، ورقمه في الفريق {p['num']}.")
        else:
            people.append(f"- {p['name']}: works as {p['job']}, lives in {p['city']}, "
                          f"is {p['age']} years old, and team number {p['num']}.")
    # الشخص المطلوب: بيتحط في مكان عشوائي في القايمة (خ12: الانتباه الكامل
    # وقت التدريب بيسمح إن الحقيقة تبقى في أي عمق في المستند)
    asked = {"name": _fill(rng, names, used), "job": rng.choice(jobs),
             "city": rng.choice(cities), "age": rng.randint(23, 55),
             "num": rng.randint(1, 99)}
    head = "بطاقة فريق المشروع:" if ar else "Project team roster:"
    body = [head] + people
    _pad_to(body, rng.choice([900, 1100, 1300, 1600]), rng, people)
    if ar:
        asked_line = (f"- {asked['name']}: شغله {asked['job']}، ساكن في {asked['city']}، "
                      f"عنده {asked['age']} سنة، ورقمه في الفريق {asked['num']}.")
    else:
        asked_line = (f"- {asked['name']}: works as {asked['job']}, lives in {asked['city']}, "
                      f"is {asked['age']} years old, and team number {asked['num']}.")
    body.insert(rng.randrange(1, len(body) + 1), asked_line)
    doc = "\n".join(body)
    if ar:
        q, kind = rng.choice([
            (f"في البطاقه دي: شغل {asked['name']} ايه", "job"),
            (f"من البطاقه: {asked['name']} ساكن فين", "city"),
            (f"في البطاقه: عمر {asked['name']} كام", "age"),
            (f"من البطاقه: رقم {asked['name']} في الفريق كام", "num"),
        ])
        ans = {"job": f"{asked['name']} شغله {asked['job']}.",
               "city": f"{asked['name']} ساكن في {asked['city']}.",
               "age": f"{asked['name']} عنده {asked['age']} سنة.",
               "num": f"رقم {asked['name']} في الفريق {asked['num']}."}[kind]
    else:
        q, kind = rng.choice([
            (f"in this roster: what is {asked['name']}'s job", "job"),
            (f"from this roster: where does {asked['name']} live", "city"),
            (f"in this roster: how old is {asked['name']}", "age"),
            (f"from this roster: what is {asked['name']}'s team number", "num"),
        ])
        ans = {"job": f"{asked['name']} works as {asked['job']}.",
               "city": f"{asked['name']} lives in {asked['city']}.",
               "age": f"{asked['name']} is {asked['age']} years old.",
               "num": f"{asked['name']}'s team number is {asked['num']}."}[kind]
    return doc, q, ans


# ============================================================
# ٢) ملف كود: دوال كتير والسؤال عن الدالة الأخيرة
# ============================================================
def make_code(rng):
    ar = rng.random() < 0.5
    used, lines = set(), []
    for _ in range(rng.randint(8, 12)):
        fname, args, ret, d_ar, d_en = _fill(rng, CODE_FUNCS, used)
        lines.append(f"def {fname}({args}):")
        lines.append(f"    {ret}")
        lines.append("")
        if rng.random() < 0.5:              # سطور تعليقات بين الدوال (حشو واقعي)
            lines.append("# الجزء ده للاستخدام الداخلي بس" if ar
                         else "# this part is for internal use only")
            lines.append("")
    fname, args, ret, d_ar, d_en = rng.choice([f for f in CODE_FUNCS if f[0] not in used])
    mod = rng.choice(["utils", "tools", "helpers", "mathx", "textx"])
    # الدالة المطلوبة في مكان عشوائي بين الدوال (مش آخرها بالضرورة)
    head = f"# {mod}.py\n\n"
    # خ12: حشو تعليقات عشان ملف الكود يفضل أطول من النافذة المنزلقة
    _pad_to(lines, rng.choice([900, 1100, 1300]), rng, CODE_FILLERS)
    pos = rng.randrange(0, len(lines) + 1)
    asked_block = [f"def {fname}({args}):", f"    {ret}", ""]
    lines[pos:pos] = asked_block
    doc = head + "\n".join(lines)
    if ar:
        q = f"بص على الكود ده: الداله {fname} بترجع ايه"
        ans = f"الداله {fname} بترجع {d_ar}."
    else:
        q = f"look at this code: what does {fname} return"
        ans = f"{fname} returns {d_en}."
    return doc, q, ans


# ============================================================
# ٣) قايمة تعليمات مرقمة — السؤال عن رقم في آخر القايمة
# ============================================================
def make_list(rng):
    ar = rng.random() < 0.5
    pool = AR_INSTR if ar else EN_INSTR
    used, items = set(), []
    n = rng.randint(26, 34)
    for _ in range(n):
        items.append(_fill(rng, pool, used))
    ask_n = rng.randint(1, n)
    head = "تعليمات التشغيل:" if ar else "Operating instructions:"
    body = [head]
    for i, t in enumerate(items, 1):
        body.append(f"{i}. {t}")
    doc = "\n".join(body)
    if ar:
        q = f"في التعليمات دي: ايه التعليمه رقم {ask_n}"
        ans = f"التعليمه رقم {ask_n} هي: {items[ask_n - 1]}."
    else:
        q = f"in these instructions: what is instruction number {ask_n}"
        ans = f"Instruction number {ask_n} is: {items[ask_n - 1]}."
    return doc, q, ans


# ============================================================
# ٤) سجل جلسة: أحداث بالساعة — السؤال عن آخر الأحداث
# ============================================================
def make_log(rng):
    ar = rng.random() < 0.5
    pool = AR_EVENTS if ar else EN_EVENTS
    used, lines = set(), []
    h, m = rng.randint(7, 9), rng.choice([0, 15, 30, 45])
    n = rng.randint(28, 38)
    for _ in range(n):
        lines.append((h, m, _fill(rng, pool, used)))
        m += rng.choice([15, 30])
        h += m // 60
        m %= 60
    ask_i = rng.randint(0, n - 1)
    ah, am, aev = lines[ask_i]
    head = "سجل الجلسه:" if ar else "Session log:"
    body = [head] + [f"الساعة {hh:02d}:{mm:02d}: {ev}" if ar else f"{hh:02d}:{mm:02d}: {ev}"
                     for hh, mm, ev in lines]
    doc = "\n".join(body)
    t = f"{ah:02d}:{am:02d}"
    if ar:
        q = f"من السجل ده: ايه اللي حصل الساعه {t}"
        ans = f"الساعه {t}: {aev}."
    else:
        q = f"from this log: what happened at {t}"
        ans = f"At {t}: {aev}."
    return doc, q, ans


MAKERS = (make_roster, make_code, make_list, make_log)
DOCS_PER_ROUND = 320          # عدد المستندات في التدريب (٨٠ من كل نوع)


def build(seed=20260, count=DOCS_PER_ROUND):
    """
    بيولّد أمثلة longdoc: كل مثال = (النص الكامل للمستند + السؤال، الرد، سطر التفكير).
    كل الأسئلة عن معلومات في آخر المستند (المنطقة اللي النافذة المنزلقة شايفها).
    """
    rng = random.Random(seed)
    out = []
    for i in range(count):
        doc, q, ans = MAKERS[i % len(MAKERS)](rng)
        # خ12: سطر تفكير ثابت — النموذج يتعلم "أمامي مستند → جاوب من جوه"
        # من غير ما يضطر ينسخ السؤال حرف بحرف (ده يخلي التعلم أسرع بكتير)
        think = "مستند | سؤال عن المستند"
        full_q = doc + "\n" + q           # نفس الصيغة اللي المحرك بيبعتها وقت التشغيل
        out.append({"question": full_q, "answer": ans, "think": think,
                    "kind": MAKERS[i % len(MAKERS)].__name__})
    return out
