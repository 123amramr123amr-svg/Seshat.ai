"""
الرسم — seshat بيرسم بإنه يكتب شبكة رموز 12 × 12، وكل رمز لون.

    .  فاضي        R أحمر      G أخضر     B أزرق     Y أصفر
    O  برتقالي     P بنفسجي    K أسود     M بمبي     N بني      W أبيض

الواجهة بتحوّل الشبكة لصورة. النموذج هو اللي بيكتب الشبكة رمز رمز (مش بياخدها
من الذاكرة)، فهو فعلًا اللي بيرسم.

تزود شكل: ضيفه في SHAPES (دالة بترجّع True للخانات الملوّنة) أو في FIXED
(رسمة ألوانها ثابتة). تزود لون: ضيفه في COLORS و PALETTE.
بعد أي إضافة لازم تدريب.
"""

N = 12
EMPTY = "."

# الرمز → اللون اللي الواجهة بترسمه بيه
PALETTE = {
    ".": "#f3efe4", "R": "#e04b3a", "G": "#3fa34d", "B": "#3b7dd8", "Y": "#f2c230",
    "O": "#ef8a2b", "P": "#8e56c9", "K": "#22252b", "M": "#f08bb4", "N": "#8a5a33",
    "W": "#ffffff",
}

# (الرمز، مذكر، مؤنث، صيغ تانية بالعربي، إنجليزي، صيغ تانية بالإنجليزي)
COLORS = [
    ("R", "احمر", "حمرا", ["حمراء"], "red", []),
    ("G", "اخضر", "خضرا", ["خضراء"], "green", []),
    ("B", "ازرق", "زرقا", ["زرقاء"], "blue", []),
    ("Y", "اصفر", "صفرا", ["صفراء"], "yellow", []),
    ("O", "برتقالي", "برتقالي", ["برتقاني", "اورنج"], "orange", []),
    ("P", "بنفسجي", "بنفسجي", ["موف"], "purple", ["violet"]),
    ("K", "اسود", "سودا", ["سوداء"], "black", []),
    ("M", "بمبي", "بمبي", ["وردي", "روز"], "pink", []),
]
DEFAULT_COLOR = "B"            # لو المستخدم مقالش لون


def _circle(x, y):
    return (x - 5.5) ** 2 + (y - 5.5) ** 2 <= 5.3 ** 2


def _ring(x, y):
    return 3.1 ** 2 <= (x - 5.5) ** 2 + (y - 5.5) ** 2 <= 5.3 ** 2


def _square(x, y):
    return 2 <= x <= 9 and 2 <= y <= 9


def _rectangle(x, y):
    return 1 <= x <= 10 and 3 <= y <= 8


def _triangle(x, y):
    return 1 <= y <= 10 and abs(x - 5.5) <= (y - 1) * 0.5 + 0.5


def _diamond(x, y):
    return abs(x - 5.5) + abs(y - 5.5) <= 5


def _plus(x, y):
    return (4 <= x <= 7 and 1 <= y <= 10) or (4 <= y <= 7 and 1 <= x <= 10)


def _heart(x, y):
    u, v = (x - 5.5) / 4.0, (5.8 - y) / 4.0
    return (u * u + v * v - 1) ** 3 - u * u * v ** 3 <= 0


def _arrow(x, y):
    shaft = 1 <= x <= 6 and 5 <= y <= 6
    head = 6 <= x <= 10 and abs(y - 5.5) <= (10 - x) + 0.5
    return shaft or head


def _line(x, y):
    return 5 <= y <= 6 and 1 <= x <= 10


# الاسم → (الدالة، الاسم العربي، مؤنث؟، أسماء عربي تانية، الاسم الإنجليزي، أسماء إنجليزي تانية)
SHAPES = {
    "circle": (_circle, "دايره", True, ["دائره", "كوره"], "circle", ["ball"]),
    "ring": (_ring, "حلقه", True, ["دايره مفرغه"], "ring", ["hollow circle"]),
    "square": (_square, "مربع", False, [], "square", []),
    "rectangle": (_rectangle, "مستطيل", False, [], "rectangle", []),
    "triangle": (_triangle, "مثلث", False, [], "triangle", []),
    "diamond": (_diamond, "معين", False, [], "diamond", []),
    "plus": (_plus, "علامه زائد", True, ["علامه بلس"], "plus sign", ["cross"]),
    "heart": (_heart, "قلب", False, [], "heart", []),
    "arrow": (_arrow, "سهم", False, [], "arrow", []),
    "line": (_line, "خط", False, [], "line", []),
}

# رسومات ألوانها ثابتة: (الاسم العربي، أسماء تانية، الاسم الإنجليزي، أسماء تانية، الشبكة)
FIXED = {
    "sun": ("شمس", ["الشمس"], "sun", ["the sun"], [
        "............",
        ".Y...Y...Y..",
        "..Y..Y..Y...",
        "....YYYY....",
        "...YYYYYY...",
        "YY.YYYYYY.YY",
        "...YYYYYY...",
        "...YYYYYY...",
        "....YYYY....",
        "..Y..Y..Y...",
        ".Y...Y...Y..",
        "............",
    ]),
    "tree": ("شجره", ["الشجره"], "tree", ["a tree"], [
        "....GGGG....",
        "...GGGGGG...",
        "..GGGGGGGG..",
        ".GGGGGGGGGG.",
        ".GGGGGGGGGG.",
        "..GGGGGGGG..",
        "...GGGGGG...",
        ".....NN.....",
        ".....NN.....",
        ".....NN.....",
        ".....NN.....",
        "....NNNN....",
    ]),
    "house": ("بيت", ["منزل", "البيت"], "house", ["a home"], [
        ".....RR.....",
        "....RRRR....",
        "...RRRRRR...",
        "..RRRRRRRR..",
        ".RRRRRRRRRR.",
        "..YYYYYYYY..",
        "..YBBYYYYY..",
        "..YBBYYNNY..",
        "..YYYYYNNY..",
        "..YYYYYNNY..",
        "..YYYYYNNY..",
        "............",
    ]),
    "egypt": ("علم مصر", ["العلم المصري"], "flag of egypt", ["egyptian flag"], [
        "............",
        "RRRRRRRRRRRR",
        "RRRRRRRRRRRR",
        "RRRRRRRRRRRR",
        "WWWWWWWWWWWW",
        "WWWWWYYWWWWW",
        "WWWWWYYWWWWW",
        "WWWWWWWWWWWW",
        "KKKKKKKKKKKK",
        "KKKKKKKKKKKK",
        "KKKKKKKKKKKK",
        "............",
    ]),
    "smile": ("وش بيضحك", ["وش مبتسم", "ابتسامه"], "smiley face", ["a happy face"], [
        "....YYYY....",
        "..YYYYYYYY..",
        ".YYYYYYYYYY.",
        ".YYKYYYYKYY.",
        "YYYKYYYYKYYY",
        "YYYYYYYYYYYY",
        "YYYYYYYYYYYY",
        "YYKYYYYYYKYY",
        ".YYKKKKKKYY.",
        ".YYYYYYYYYY.",
        "..YYYYYYYY..",
        "....YYYY....",
    ]),
    "pyramid": ("هرم", ["الهرم", "اهرامات"], "pyramid", ["the pyramids"], [
        "BBBBBBBBBBBB",
        "BBBBBBBBBYYB",
        "BBBBBOBBBYYB",
        "BBBBBOOBBBBB",
        "BBBBOOOOBBBB",
        "BBBBOOOOBBBB",
        "BBBOOOOOOBBB",
        "BBBOOOOOOBBB",
        "BBOOOOOOOOBB",
        "BBOOOOOOOOBB",
        "BOOOOOOOOOOB",
        "YYYYYYYYYYYY",
    ]),
}

# تركيبات (شكل، لون) النموذج مبيتدربش عليها خالص — عشان نختبر هل اتعلم "يرسم"
# ولا حفظ الرسومات بس.
HELD_OUT = {
    ("circle", "P"), ("square", "O"), ("triangle", "M"), ("heart", "G"),
    ("diamond", "R"), ("plus", "Y"), ("ring", "K"), ("arrow", "G"),
}

AR_TEMPLATES = ["ارسم {x}", "ارسملي {x}", "عايز صوره {x}", "اعمل رسمه {x}", "صوره {x}",
                "ممكن ترسم {x}", "انشئ صوره {x}", "عايز رسمه {x}"]
EN_TEMPLATES = ["draw a {x}", "draw {x}", "make an image of a {x}", "create a picture of a {x}",
                "generate a {x}", "a {x} image", "can you draw a {x}", "picture of a {x}"]


def grid_for(shape, code):
    fn = SHAPES[shape][0]
    return ["".join(code if fn(x, y) else EMPTY for x in range(N)) for y in range(N)]


def block(rows):
    return "```draw\n" + "\n".join(rows) + "\n```"


def _colored(shape, color):
    """بيرجع (أسئلة عربي، رد عربي، أسئلة إنجليزي، رد إنجليزي)"""
    fn, ar, fem, ar_alt, en, en_alt = SHAPES[shape]
    code, masc, femc, c_alt, en_c, en_c_alt = color
    main = femc if fem else masc
    names = [f"{ar} {main}"]                                       # الصيغة الأساسية أول واحدة
    names += [f"{a} {main}" for a in ar_alt]
    names += [f"{ar} {c}" for c in ({masc, femc, *c_alt} - {main})]
    names += [f"{ar} لونه {masc}", f"{ar} باللون ال{masc}"]
    en_names = [f"{en_c} {en}"] + [f"{en_c} {e}" for e in en_alt] + [f"{c} {en}" for c in en_c_alt]
    if code == DEFAULT_COLOR:                                      # من غير لون
        names += [ar] + ar_alt
        en_names += [en] + en_alt

    rows = grid_for(shape, code)
    ar_qs = [t.format(x=n) for n in names for t in AR_TEMPLATES]
    en_qs = [t.format(x=n) for n in en_names for t in EN_TEMPLATES]
    return (ar_qs, f"دي رسمة {ar} {main}:\n{block(rows)}",
            en_qs, f"Here is a {en_c} {en}:\n{block(rows)}")


def build(include_held_out=False):
    """
    بيرجع قايمة (أسئلة، رد). بـ include_held_out=True بيرجع التركيبات المتشالة للاختبار بس.
    """
    out = []
    for shape in SHAPES:
        for color in COLORS:
            held = (shape, color[0]) in HELD_OUT
            if held != include_held_out:
                continue
            ar_qs, ar_a, en_qs, en_a = _colored(shape, color)
            out += [(ar_qs, ar_a), (en_qs, en_a)]
    if not include_held_out:
        for ar, ar_alt, en, en_alt, rows in FIXED.values():
            assert len(rows) == N and all(len(r) == N for r in rows), ar
            ar_qs = [t.format(x=n) for n in [ar] + ar_alt for t in AR_TEMPLATES]
            en_qs = [t.format(x=n) for n in [en] + en_alt for t in EN_TEMPLATES]
            out += [(ar_qs, f"دي رسمة {ar}:\n{block(rows)}"),
                    (en_qs, f"Here is a {en}:\n{block(rows)}")]
    return out


def parse_grid(text):
    """بيطلّع الشبكة من الرد. بيرجع None لو الرسمة مش مظبوطة (مش 12 × 12 أو فيها رمز غريب)."""
    if "```draw" not in text:
        return None
    body = text.split("```draw", 1)[1].split("```", 1)[0].strip("\n")
    rows = body.split("\n")
    if len(rows) != N or any(len(r) != N or set(r) - set(PALETTE) for r in rows):
        return None
    return rows


def what_was_asked(question):
    """
    بيدوّر في الجملة على شكل ولون معروفين. بيرجع (الشكل أو None، رمز اللون أو None).
    بنستخدمها عشان نتأكد إن النموذج رسم اللي اتطلب منه مش حاجة تانية.
    """
    words = " " + question + " "
    shape, best = None, 0
    names = [(k, [v[1]] + v[3] + [v[4]] + v[5]) for k, v in SHAPES.items()]
    names += [(k, [v[0]] + v[1] + [v[2]] + v[3]) for k, v in FIXED.items()]
    for key, alts in names:
        for name in alts:
            if f" {name} " in words and len(name) > best:      # أطول اسم يكسب (حلقه × دايره مفرغه)
                shape, best = key, len(name)
    color = None
    for code, masc, femc, alt, en, en_alt in COLORS:
        for name in [masc, femc, en] + alt + en_alt:
            if f" {name} " in words or f" ال{name} " in words:
                color = code
    return shape, color


if __name__ == "__main__":
    for shape in SHAPES:
        print(shape)
        print("\n".join(grid_for(shape, "#")))
    print(len(build()), "رسمة للتدريب |", len(build(True)), "للاختبار")
