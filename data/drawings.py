"""
الرسم — seshat بيرسم بنفسه: بيكتب الصورة حرف حرف بصيغة مضغوطة، والمحرك بيفكها لشبكة
16 × 16 مربع ملون والواجهة بتعرضها صورة.

الصيغة اللي النموذج بيكتبها (جوه بلوك ```draw):

    #=R              ← اللون الأساسي للرسمة (# في السطور اللي تحت = الرمز ده)
    a16.             ← كل سطر بيبدأ بحرف رقمه (a أول سطر … p آخر سطر)
    b5.6#5.          ← 5 فاضي، 6 باللون الأساسي، 5 فاضي  (العدد 1 مبيتكتبش)
    c=               ← نفس السطر اللي فوق
    d4.#K4#K#4.      ← ...

ليه الصيغة دي؟ النموذج بيقرا ويكتب حروف، وكان بيغلط في عدّ الخانات لما يكتب 12 حرف ورا بعض
(السطر يطلع 11 أو 13). هنا بيكتب العدد رقم، والسطر المتكرر علامة واحدة، واللون الأساسي
مرة واحدة فوق — فالرسمة أقصر مرتين ونص تقريبًا، ونفس الشكل بيتكتب بنفس الحروف لأي لون.
وحرف رقم السطر بيخلي النموذج عارف هو في أنهي سطر من غير ما يعدّ السطور اللي فاتت.

الرموز:
    .  فاضي     R أحمر    G أخضر   B أزرق   Y أصفر    O برتقالي   P بنفسجي
    K  أسود     M بمبي    N بني    W أبيض   S رمادي   C سماوي
    #  اللون الأساسي (بيتحدد في أول سطر)

تزود رسمة: ضيف دالة في OBJECTS بترسم على Canvas، وشوفها بـ  python data/drawings.py --sheet out.png
بعد أي إضافة لازم تدريب.
"""

import re

N = 16
ROWS = "abcdefghijklmnop"      # حرف رقم السطر
EMPTY = "."
MAIN = "#"

# الرمز → اللون اللي الواجهة بترسمه بيه
PALETTE = {
    ".": "#ece7da", "R": "#e04b3a", "G": "#3fa34d", "B": "#3b7dd8", "Y": "#f2c230",
    "O": "#ef8a2b", "P": "#8e56c9", "K": "#22252b", "M": "#f08bb4", "N": "#8a5a33",
    "W": "#ffffff", "S": "#9aa0a6", "C": "#7cc4f2",
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
    ("W", "ابيض", "بيضا", ["بيضاء"], "white", []),
    ("N", "بني", "بني", [], "brown", []),
    ("S", "رمادي", "رمادي", ["رصاصي"], "grey", ["gray"]),
]
COLOR_BY_CODE = {c[0]: c for c in COLORS}


# ============================================================
# لوحة رسم صغيرة بنبني بيها الرسومات (وقت تجهيز البيانات بس)
# ============================================================
class Canvas:
    def __init__(self):
        self.g = [[EMPTY] * N for _ in range(N)]

    def dot(self, x, y, c):
        if 0 <= x < N and 0 <= y < N:
            self.g[y][x] = c
        return self

    def dots(self, pts, c):
        for x, y in pts:
            self.dot(x, y, c)
        return self

    def rect(self, x0, y0, x1, y1, c):
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                self.dot(x, y, c)
        return self

    def ell(self, cx, cy, rx, ry, c):
        for y in range(N):
            for x in range(N):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                    self.g[y][x] = c
        return self

    def tri(self, cx, y0, y1, half, c):
        """مثلث راسه فوق عند (cx, y0) وقاعدته عند y1 بنص عرض half"""
        for y in range(y0, y1 + 1):
            w = half * (y - y0) / max(1, y1 - y0)
            for x in range(N):
                if abs(x - cx) <= w + 0.01:
                    self.dot(x, y, c)
        return self

    def tri_down(self, cx, y0, y1, half, c):
        """مثلث مقلوب: قاعدته فوق عند y0 وراسه تحت عند y1"""
        for y in range(y0, y1 + 1):
            w = half * (y1 - y) / max(1, y1 - y0)
            for x in range(N):
                if abs(x - cx) <= w + 0.01:
                    self.dot(x, y, c)
        return self

    def paint(self, rows, x0=0, y0=0):
        """بيحط رسمة مكتوبة بالحروف (المسافة = سيب اللي تحتها)"""
        for dy, row in enumerate(rows):
            for dx, ch in enumerate(row):
                if ch != " ":
                    self.dot(x0 + dx, y0 + dy, ch)
        return self

    def rows(self):
        return ["".join(r) for r in self.g]


M_ = MAIN
MID = (N - 1) / 2          # 7.5


def _circle(c):
    c.ell(MID, MID, 6.6, 6.6, M_)


def _ring(c):
    c.ell(MID, MID, 6.6, 6.6, M_).ell(MID, MID, 3.6, 3.6, EMPTY)


def _square(c):
    c.rect(3, 3, 12, 12, M_)


def _rectangle(c):
    c.rect(1, 4, 14, 11, M_)


def _triangle(c):
    c.tri(MID, 2, 13, 7, M_)


def _diamond(c):
    for y in range(N):
        for x in range(N):
            if abs(x - MID) + abs(y - MID) <= 7:
                c.dot(x, y, M_)


def _plus(c):
    c.rect(6, 1, 9, 14, M_).rect(1, 6, 14, 9, M_)


def _arrow(c):
    c.rect(1, 6, 9, 9, M_)
    for i in range(7):
        c.rect(9 + i, 1 + i, 9 + i, 14 - i, M_)


def _line(c):
    c.rect(1, 7, 14, 8, M_)


def _heart(c):
    c.ell(4.5, 5, 3.4, 3.4, M_).ell(10.5, 5, 3.4, 3.4, M_)
    c.rect(2, 5, 13, 7, M_)
    c.tri_down(MID, 7, 14, 6.5, M_)


def _star(c):
    c.tri(MID, 1, 6, 2.5, M_)
    c.rect(1, 6, 14, 7, M_).rect(2, 8, 13, 8, M_).rect(3, 9, 12, 9, M_)
    c.rect(4, 10, 11, 11, M_)
    c.rect(3, 12, 6, 12, M_).rect(9, 12, 12, 12, M_)
    c.rect(3, 13, 5, 13, M_).rect(10, 13, 12, 13, M_)
    c.rect(2, 14, 4, 14, M_).rect(11, 14, 13, 14, M_)


def _cat(c):
    c.tri(4, 1, 3, 1.5, M_).tri(11, 1, 3, 1.5, M_)                  # الودان
    c.ell(MID, 6, 5.2, 3.6, M_)                                      # الراس
    c.dots([(5, 5), (10, 5)], "K")                                   # العيون
    c.dots([(7, 7), (8, 7)], "M")                                    # الأنف
    c.rect(4, 10, 11, 14, M_)                                        # الجسم
    c.dots([(12, 13), (13, 12), (13, 11), (14, 10), (12, 14), (13, 13)], M_)   # الديل
    c.dots([(5, 15), (6, 15), (9, 15), (10, 15)], M_)                # الرجلين


def _dog(c):
    c.rect(4, 2, 11, 8, M_)                                          # الراس
    c.rect(2, 2, 3, 7, "K").rect(12, 2, 13, 7, "K")                  # ودان مدلدلة
    c.dots([(6, 4), (9, 4)], "K")
    c.rect(6, 6, 9, 8, "W").dots([(7, 6), (8, 6)], "K")              # البوز والأنف
    c.rect(4, 9, 11, 13, M_)                                         # الجسم
    c.rect(4, 14, 5, 15, M_).rect(10, 14, 11, 15, M_)                # الرجلين
    c.dots([(12, 10), (13, 9), (14, 8)], M_)                         # الديل


def _fish(c):
    c.ell(6, 7.5, 5.4, 3.8, M_)                                      # الجسم
    for i in range(4):                                               # الديل
        c.rect(11 + i, 7 - i, 11 + i, 8 + i, M_)
    c.dot(3, 6, "K")                                                 # العين
    c.dots([(6, 6), (6, 7), (6, 8), (6, 9)], "W")                    # خط


def _bird(c):
    c.ell(6.5, 9, 4.6, 3.6, M_)                                      # الجسم
    c.ell(10.5, 5, 2.6, 2.6, M_)                                     # الراس
    c.dot(11, 4, "K")
    c.dots([(13, 5), (14, 5), (13, 6)], "O")                         # المنقار
    c.dots([(1, 7), (2, 8), (1, 8), (2, 9)], M_)                     # الديل
    c.rect(5, 8, 8, 9, "W")                                          # الجناح
    c.dots([(6, 13), (6, 14), (8, 13), (8, 14)], "O")                # الرجلين
    c.dots([(5, 15), (6, 15), (8, 15), (9, 15)], "O")


def _duck(c):
    c.ell(7, 10.5, 5.6, 3.4, M_)                                     # الجسم
    c.ell(10.5, 4.5, 2.7, 2.7, M_)                                   # الراس
    c.rect(9, 6, 11, 8, M_)                                          # الرقبة
    c.dot(11, 4, "K")
    c.dots([(13, 5), (14, 5), (13, 6), (14, 6)], "O")                # المنقار
    c.rect(0, 14, 15, 15, "C")                                       # الميه


def _butterfly(c):
    c.ell(3.5, 5, 3.3, 3.6, M_).ell(11.5, 5, 3.3, 3.6, M_)           # جناحين فوق
    c.ell(4, 11, 2.7, 2.7, M_).ell(11, 11, 2.7, 2.7, M_)             # جناحين تحت
    c.dots([(3, 5), (12, 5), (4, 11), (11, 11)], "W")
    c.rect(7, 3, 8, 13, "K")                                         # الجسم
    c.dots([(6, 1), (9, 1), (6, 2), (9, 2)], "K")                    # قرون الاستشعار


def _tree(c):
    c.ell(MID, 5.5, 5.6, 5, M_)                                      # الورق
    c.rect(6, 10, 9, 14, "N")                                        # الجذع
    c.rect(3, 15, 12, 15, "N")


def _palm(c):
    c.paint(["  ###    ###  ",
             " #####  ##### ",
             "##  ######  ##",
             "#    ####    #",
             "    ######    ",
             "   ## ## ##   ",
             "  ##  ##  ##  ",
             "  #   ##   #  "], 1, 0)
    c.rect(7, 4, 8, 14, "N")                                         # الجذع
    c.rect(4, 15, 11, 15, "Y")                                       # الرمل


def _flower(c):
    c.ell(MID, 2.5, 2, 2, M_).ell(MID, 8.5, 2, 2, M_)
    c.ell(4.5, 5.5, 2, 2, M_).ell(10.5, 5.5, 2, 2, M_)               # البتلات
    c.rect(6, 4, 9, 7, "Y")                                          # القلب
    c.rect(7, 10, 8, 15, "G")                                        # الساق
    c.dots([(5, 11), (6, 12), (5, 12), (10, 12), (9, 13), (10, 13)], "G")   # ورقتين


def _apple(c):
    c.ell(MID, 9, 5.6, 5.2, M_)
    c.rect(7, 3, 8, 3, EMPTY)
    c.rect(7, 1, 7, 4, "N")                                          # العود
    c.dots([(8, 1), (9, 1), (10, 1), (9, 2), (10, 2), (9, 0)], "G")   # الورقة
    c.dots([(4, 7), (4, 8), (5, 6)], "W")                            # لمعة


def _carrot(c):
    c.tri_down(MID, 5, 15, 3.6, M_)
    c.rect(4, 4, 11, 5, M_)
    c.paint([" G  GG  G ",
             " GG GG GG ",
             "  GGGGGG  ",
             "   GGGG   "], 3, 0)


def _mushroom(c):
    c.ell(MID, 6.5, 6.8, 4.6, M_)                                    # الراس
    c.rect(0, 7, 15, 11, EMPTY)
    c.rect(1, 7, 14, 8, M_)
    c.dots([(4, 4), (5, 4), (4, 5), (10, 3), (11, 3), (10, 4), (7, 6), (8, 6), (12, 6)], "W")
    c.rect(5, 9, 10, 14, "W")                                        # الساق
    c.rect(4, 15, 11, 15, "G")


def _icecream(c):
    c.ell(MID, 4.5, 4.6, 4, M_)                                      # الكورة
    c.rect(3, 6, 12, 7, M_)
    c.tri_down(MID, 8, 15, 4, "N")                                   # البسكوت
    c.dots([(6, 9), (8, 9), (7, 11)], "Y")
    c.dot(7, 0, "R").dot(8, 0, "R")                                  # كرزة


def _cup(c):
    c.rect(2, 4, 10, 13, M_)
    c.rect(3, 4, 9, 4, "N")                                          # القهوة
    c.rect(11, 6, 13, 6, M_).rect(13, 7, 13, 10, M_).rect(11, 11, 13, 11, M_)   # الودن
    c.rect(1, 14, 11, 14, "S")                                       # الطبق
    c.dots([(4, 1), (5, 2), (4, 3), (8, 1), (7, 2), (8, 3)], "S")    # البخار


def _house(c):
    c.tri(MID, 1, 6, 7.5, "R")                                       # السقف
    c.rect(2, 7, 13, 15, M_)                                         # الحيطان
    c.rect(7, 11, 9, 15, "N")                                        # الباب
    c.rect(3, 9, 5, 11, "C").rect(11, 9, 12, 11, "C")                # الشبابيك
    c.rect(11, 1, 12, 3, "N")                                        # المدخنة


def _car(c):
    c.rect(4, 4, 11, 7, M_)                                          # الكابينة
    c.rect(5, 5, 7, 7, "C").rect(9, 5, 10, 7, "C")                   # الإزاز
    c.rect(1, 8, 14, 11, M_)                                         # الجسم
    c.dot(14, 9, "Y").dot(1, 9, "R")                                 # الكشافات
    c.ell(4, 12.5, 1.6, 1.6, "K").ell(11, 12.5, 1.6, 1.6, "K")       # العجل
    c.dot(4, 12, "S").dot(11, 12, "S")


def _boat(c):
    c.rect(7, 1, 7, 10, "N")                                         # الصاري
    c.tri(8, 1, 9, 5, "W")
    c.rect(0, 1, 7, 9, EMPTY).rect(7, 1, 7, 10, "N")                 # شراع يمين بس
    c.paint(["##############",
             " ############ ",
             "  ##########  "], 1, 10)                                 # المركب
    c.rect(0, 13, 15, 15, "C")                                       # الميه
    c.dots([(2, 13), (3, 13), (8, 14), (9, 14), (13, 13)], "B")


def _rocket(c):
    c.tri(MID, 0, 3, 2.5, "R")                                       # الراس
    c.rect(5, 4, 10, 11, M_)                                         # الجسم
    c.rect(7, 5, 8, 6, "C")                                          # الشباك
    c.dots([(4, 9), (4, 10), (4, 11), (3, 10), (3, 11), (11, 9), (11, 10), (11, 11),
            (12, 10), (12, 11)], "R")                                # الجناحات
    c.rect(6, 12, 9, 13, "O").rect(7, 14, 8, 15, "Y")                # النار


def _balloon(c):
    c.ell(MID, 5.5, 5, 5.4, M_)
    c.dots([(5, 3), (5, 4), (6, 2)], "W")                            # لمعة
    c.dots([(7, 11), (8, 11)], M_)
    c.dots([(7, 12), (8, 13), (7, 14), (8, 15)], "K")                # الخيط


def _umbrella(c):
    c.ell(MID, 7, 7.4, 6, M_)
    c.rect(0, 8, 15, 15, EMPTY)
    c.dots([(2, 8), (5, 8), (10, 8), (13, 8)], EMPTY)
    c.rect(7, 8, 7, 13, "N")                                         # العصاية
    c.dots([(7, 14), (8, 14), (9, 14), (9, 13)], "N")
    c.dot(7, 0, "N")


def _sun(c):
    c.ell(MID, MID, 4.2, 4.2, M_)
    c.dots([(7, 0), (8, 0), (7, 1), (8, 1), (7, 14), (8, 14), (7, 15), (8, 15),
            (0, 7), (0, 8), (1, 7), (1, 8), (14, 7), (14, 8), (15, 7), (15, 8),
            (2, 2), (3, 3), (12, 12), (13, 13), (13, 2), (12, 3), (2, 13), (3, 12)], M_)
    c.dots([(6, 7), (9, 7)], "K")
    c.dots([(6, 9), (7, 10), (8, 10), (9, 9)], "K")


def _moon(c):
    c.ell(7, MID, 6.4, 6.4, M_)
    c.ell(10.5, 6, 5.4, 5.4, EMPTY)
    c.dots([(13, 2), (12, 12), (14, 8)], "W")                        # نجوم


def _cloud(c):
    c.ell(4.5, 9, 3.6, 3, M_).ell(8, 6.5, 3.8, 3.8, M_).ell(11.5, 9, 3.6, 3, M_)
    c.rect(3, 9, 12, 11, M_)


def _mountain(c):
    c.tri(11, 7, 14, 4, "S")                                         # جبل صغير ورا
    c.tri(6, 3, 14, 6, M_)
    c.tri(6, 3, 5, 1.2, "W")                                         # التلج
    c.rect(0, 15, 15, 15, "G")


def _pyramid(c):
    c.tri(MID, 2, 13, 7.5, M_)
    for y in range(4, 14, 3):                                        # صفوف الحجارة
        for x in range(N):
            if c.g[y][x] == M_ and x % 3 == y % 2:
                c.g[y][x] = "N"
    c.rect(0, 14, 15, 15, "Y")                                       # الرمل


def _smiley(c):
    c.ell(MID, MID, 6.6, 6.6, M_)
    c.rect(5, 5, 5, 6, "K").rect(10, 5, 10, 6, "K")
    c.dots([(4, 9), (5, 10), (6, 11), (7, 11), (8, 11), (9, 11), (10, 10), (11, 9)], "K")


def _robot(c):
    c.rect(7, 0, 8, 1, "S").dot(7, 0, "R").dot(8, 0, "R")            # الإريال
    c.rect(4, 2, 11, 7, M_)                                          # الراس
    c.rect(5, 4, 6, 5, "W").rect(9, 4, 10, 5, "W")                   # العيون
    c.rect(6, 7, 9, 7, "K")                                          # البق
    c.rect(3, 9, 12, 13, M_)                                         # الجسم
    c.rect(6, 10, 9, 11, "Y")
    c.rect(1, 9, 2, 12, "S").rect(13, 9, 14, 12, "S")                # الدراعات
    c.rect(4, 14, 6, 15, "S").rect(9, 14, 11, 15, "S")               # الرجلين


def _crown(c):
    c.paint(["#     ##     #",
             "#     ##     #",
             "##   ####   ##",
             "### ###### ###",
             "##############",
             "##############",
             "##############",
             "##############"], 1, 4)
    c.dots([(1, 3), (14, 3), (7, 3), (8, 3)], "R")
    c.dots([(4, 10), (7, 10), (8, 10), (11, 10)], "B")
    c.rect(1, 12, 14, 12, "N")


def _book(c):
    c.rect(2, 3, 13, 13, M_)
    c.rect(7, 3, 8, 13, "K")                                         # الكعب
    c.rect(3, 4, 6, 12, "W").rect(9, 4, 12, 12, "W")                 # الصفحات
    for y in (6, 8, 10):
        c.rect(4, y, 5, y, "S").rect(10, y, 11, y, "S")


def _bulb(c):
    c.ell(MID, 5.5, 4.6, 5, M_)
    c.rect(6, 10, 9, 11, M_)
    c.rect(6, 12, 9, 13, "S").rect(7, 14, 8, 14, "K")                # القاعدة
    c.dots([(5, 3), (5, 4)], "W")


def _key(c):
    c.ell(4, MID, 3.4, 3.4, M_).ell(4, MID, 1.3, 1.3, EMPTY)
    c.rect(7, 7, 14, 8, M_)
    c.rect(11, 9, 11, 10, M_).rect(13, 9, 14, 11, M_)


def _egypt(c):
    c.rect(1, 3, 14, 5, M_)
    c.rect(1, 6, 14, 8, "W")
    c.rect(1, 9, 14, 11, "K")
    c.rect(7, 6, 8, 8, "Y")                                          # النسر


def _ball(c):
    c.ell(MID, MID, 6.6, 6.6, M_)
    c.paint(["  KK  ",
             " KKKK ",
             "KKKKKK",
             " KKKK ",
             "  KK  "], 5, 5)
    c.dots([(2, 4), (3, 3), (12, 3), (13, 4), (3, 12), (12, 12)], "K")


# المفتاح → (الدالة، الاسم العربي، مؤنث؟، أسماء تانية، الإنجليزي، أسماء تانية، اللون الافتراضي)
OBJECTS = {
    # أشكال هندسية
    "circle": (_circle, "دايره", True, ["دائره"], "circle", [], "B"),
    "ring": (_ring, "حلقه", True, ["دايره مفرغه"], "ring", ["hollow circle"], "B"),
    "square": (_square, "مربع", False, [], "square", [], "B"),
    "rectangle": (_rectangle, "مستطيل", False, [], "rectangle", [], "B"),
    "triangle": (_triangle, "مثلث", False, [], "triangle", [], "B"),
    "diamond": (_diamond, "معين", False, [], "diamond", [], "B"),
    "plus": (_plus, "علامه زائد", True, ["علامه بلس"], "plus sign", ["cross"], "B"),
    "arrow": (_arrow, "سهم", False, [], "arrow", [], "B"),
    "line": (_line, "خط", False, [], "line", [], "B"),
    "heart": (_heart, "قلب", False, [], "heart", [], "R"),
    "star": (_star, "نجمه", True, ["نجم"], "star", [], "Y"),
    # حيوانات
    "cat": (_cat, "قطه", True, ["قط", "بسه"], "cat", ["kitten"], "O"),
    "dog": (_dog, "كلب", False, ["جرو"], "dog", ["puppy"], "N"),
    "fish": (_fish, "سمكه", True, ["سمك"], "fish", [], "B"),
    "bird": (_bird, "عصفوره", True, ["عصفور", "طاير"], "bird", [], "Y"),
    "duck": (_duck, "بطه", True, ["بط"], "duck", [], "Y"),
    "butterfly": (_butterfly, "فراشه", True, [], "butterfly", [], "P"),
    # نباتات وأكل
    "tree": (_tree, "شجره", True, [], "tree", [], "G"),
    "palm": (_palm, "نخله", True, [], "palm tree", ["palm"], "G"),
    "flower": (_flower, "ورده", True, ["زهره"], "flower", ["rose"], "R"),
    "apple": (_apple, "تفاحه", True, ["تفاح"], "apple", [], "R"),
    "carrot": (_carrot, "جزره", True, ["جزر"], "carrot", [], "O"),
    "mushroom": (_mushroom, "عش الغراب", False, ["مشروم"], "mushroom", [], "R"),
    "icecream": (_icecream, "ايس كريم", False, ["جيلاتي"], "ice cream", [], "M"),
    "cup": (_cup, "فنجان", False, ["كوبايه", "كوب"], "cup", ["mug"], "B"),
    # حاجات
    "house": (_house, "بيت", False, ["منزل"], "house", ["home"], "Y"),
    "car": (_car, "عربيه", True, ["سياره"], "car", [], "R"),
    "boat": (_boat, "مركب", True, ["سفينه"], "boat", ["ship"], "N"),
    "rocket": (_rocket, "صاروخ", False, [], "rocket", [], "W"),
    "balloon": (_balloon, "بالونه", True, ["بالون"], "balloon", [], "R"),
    "umbrella": (_umbrella, "شمسيه", True, ["مظله"], "umbrella", [], "P"),
    "robot": (_robot, "روبوت", False, [], "robot", [], "B"),
    "crown": (_crown, "تاج", False, [], "crown", [], "Y"),
    "book": (_book, "كتاب", False, [], "book", [], "G"),
    "bulb": (_bulb, "لمبه", True, ["مصباح"], "light bulb", ["bulb", "lamp"], "Y"),
    "key": (_key, "مفتاح", False, [], "key", [], "Y"),
    "ball": (_ball, "كوره", True, [], "ball", ["football"], "W"),
    # طبيعة ورموز
    "sun": (_sun, "شمس", True, [], "sun", [], "Y"),
    "moon": (_moon, "قمر", False, ["هلال"], "moon", ["crescent"], "Y"),
    "cloud": (_cloud, "سحابه", True, ["سحاب", "غيمه"], "cloud", [], "C"),
    "mountain": (_mountain, "جبل", False, [], "mountain", [], "N"),
    "pyramid": (_pyramid, "هرم", False, ["اهرامات"], "pyramid", [], "O"),
    "smiley": (_smiley, "وش مبتسم", False, ["وش بيضحك", "ابتسامه"], "smiley face", ["smiley", "smile"], "Y"),
    "egypt": (_egypt, "علم مصر", False, [], "flag of egypt", ["egypt flag", "egyptian flag"], "R"),
}
# رسومات لونها الأساسي ثابت (مبتتطلبش بلون تاني)
FIXED_COLOR = {"egypt"}

# تركيبات (رسمة، لون) متشالة من التدريب عشان نقيس: النموذج شاف الرسمة بألوان تانية وشاف
# اللون على رسومات تانية، بس عمره ما شاف الاتنين مع بعض.
HELD_OUT = {
    ("circle", "P"), ("square", "O"), ("triangle", "M"), ("heart", "G"), ("diamond", "R"),
    ("plus", "Y"), ("ring", "K"), ("arrow", "G"), ("star", "B"), ("cat", "P"), ("dog", "W"),
    ("fish", "R"), ("bird", "B"), ("tree", "Y"), ("flower", "P"), ("car", "B"), ("house", "M"),
    ("balloon", "G"), ("cup", "R"), ("robot", "S"), ("apple", "G"), ("butterfly", "O"),
    ("boat", "K"), ("umbrella", "Y"), ("book", "N"), ("moon", "W"),
}

AR_TEMPLATES = ["ارسم {x}", "ارسملي {x}", "عايز صوره {x}", "اعمل رسمه {x}", "صوره {x}",
                "ممكن ترسم {x}", "انشئ صوره {x}", "عايز رسمه {x}", "اعملي صوره {x}"]
EN_TEMPLATES = ["draw a {x}", "draw {x}", "make an image of a {x}", "create a picture of a {x}",
                "generate a {x}", "a {x} image", "can you draw a {x}", "picture of a {x}"]

_DESIGNS = {}


def design(key):
    """الرسمة بالرمز # مكان اللون الأساسي — 16 سطر كل سطر 16 رمز"""
    if key not in _DESIGNS:
        c = Canvas()
        OBJECTS[key][0](c)
        _DESIGNS[key] = c.rows()
    return _DESIGNS[key]


def contrast(code):
    """لون التفاصيل لو كانت بنفس اللون الأساسي (عين سودا في قطة سودا مش هتبان)"""
    return "W" if code != "W" else "K"


def colorize(rows, code):
    """بيحط اللون الأساسي مكان #. والتفاصيل اللي بنفس اللون بتتقلب للون يبان."""
    other = contrast(code)
    return ["".join(code if ch == MAIN else (other if ch == code else ch) for ch in row)
            for row in rows]


def grid_for(key, code=None):
    """الشبكة النهائية (16 × 16 رمز لون) لرسمة بلون أساسي معين"""
    return colorize(design(key), code or OBJECTS[key][6])


# ============================================================
# الصيغة المضغوطة: كتابة وقراية
# ============================================================
def _rle(row):
    out, i = [], 0
    while i < len(row):
        j = i
        while j < len(row) and row[j] == row[i]:
            j += 1
        out.append((str(j - i) if j - i > 1 else "") + row[i])
        i = j
    return "".join(out)


def encode(rows, code):
    """رسمة بالرمز # + اللون الأساسي → النص اللي النموذج بيكتبه"""
    lines, prev = [f"{MAIN}={code}"], None
    for label, row in zip(ROWS, rows):
        lines.append(label + ("=" if row == prev else _rle(row)))
        prev = row
    return "\n".join(lines)


def block(text):
    return "```draw\n" + (text if isinstance(text, str) else "\n".join(text)) + "\n```"


_RUN = re.compile(r"(\d{0,2})([^\d\s])")


def decode(text, loose=True):
    """
    بيفك اللي النموذج كتبه لشبكة ألوان. بيرجع (الشبكة، رمز اللون الأساسي، عدد السطور اللي
    اتظبطت) أو None لو الرسمة بايظة.
    loose=False: لازم كل حاجة مظبوطة (16 سطر بالترتيب، ومجموع كل سطر 16).
    loose=True بيسامح في غلطات العدّ الصغيرة من غير ما يغيّر أي خانة النموذج كتبها:
      • السطر اللي ناقص/زايد خانة أو اتنين بيتكمل فاضي أو بيتقص من اليمين.
      • كل سطر بيتحط في مكانه حسب حرفه، ولو فيه سطر أو اتنين متكتبوش بيفضلوا فاضيين.
    """
    if "```draw" in text:
        text = text.split("```draw", 1)[1].split("```", 1)[0]
    lines = [ln.strip() for ln in text.strip("\n").split("\n")]
    if not lines or not re.fullmatch(r"#=[A-Z]", lines[0]) or lines[0][2] not in PALETTE:
        return None
    code = lines[0][2]
    slack = 2 if loose else 0
    placed, fixed, prev = {}, 0, None
    for i, ln in enumerate(lines[1:]):
        if not ln or ln[0] not in ROWS:
            return None
        at, body = ROWS.index(ln[0]), ln[1:]
        if not loose and at != i:
            return None                                    # السطور لازم تيجي بالترتيب
        if at in placed:
            return None                                    # نفس السطر اتكتب مرتين
        if body == "=":
            if prev is None:
                return None
            row = prev
        else:
            if not body or _RUN.sub("", body):
                return None                                # فيه حاجة مش (عدد + رمز)
            row = "".join(sym * int(n or 1) for n, sym in _RUN.findall(body))
            if set(row) - set(PALETTE) - {MAIN} or abs(len(row) - N) > slack:
                return None
            if len(row) != N:
                fixed += 1
                row = (row + EMPTY * N)[:N]
        placed[at] = prev = row
    missing = N - len(placed)
    if missing > slack:
        return None
    fixed += missing
    rows = [placed.get(i, EMPTY * N) for i in range(N)]
    return colorize(rows, code), code, fixed


def parse_grid(text):
    """القراية الصارمة: الرسمة لازم تكون مظبوطة بالظبط. بيرجع الشبكة أو None."""
    got = decode(text, loose=False)
    return got[0] if got else None


def parse_grid_loose(text):
    """القراية المتسامحة: بيرجع (الشبكة، عدد السطور اللي اتظبطت) أو None."""
    got = decode(text, loose=True)
    return (got[0], got[2]) if got else None


def full_block(grids):
    """شبكة أو أكتر جنب بعض (بينهم عمود فاضي) → بلوك ```draw بالألوان الكاملة للواجهة"""
    rows = ["".join(parts) for parts in zip(*[[r + EMPTY for r in g] for g in grids])]
    return "```draw\n" + "\n".join(r[:-1] for r in rows) + "\n```"


def read_full(text):
    """بيقرا بلوك الألوان الكاملة اللي المحرك بيبعته للواجهة → قايمة سطور، أو None"""
    if "```draw" not in text:
        return None
    return text.split("```draw", 1)[1].split("```", 1)[0].strip("\n").split("\n")


# ============================================================
# الأسماء والأسئلة
# ============================================================
def color_word(code, feminine, arabic=True):
    c = COLOR_BY_CODE[code]
    return (c[2] if feminine else c[1]) if arabic else c[4]


def title(key, code, arabic=True):
    """اسم الرسمة في الرد: «قطه سودا» / “black cat”"""
    fn, ar, fem, ar_alt, en, en_alt, default = OBJECTS[key]
    if key in FIXED_COLOR:
        return ar if arabic else en
    return f"{ar} {color_word(code, fem)}" if arabic else f"{color_word(code, fem, False)} {en}"


def canonical(key, code, arabic=True):
    """الصيغة الأساسية للطلب — اللي المحرك بيسأل بيها النموذج لما يرسم حاجة جوه مشهد"""
    return ("ارسم " if arabic else "draw a ") + title(key, code, arabic)


def answer(key, code, arabic=True):
    head = f"دي رسمة {title(key, code)}:" if arabic else f"Here is a {title(key, code, False)}:"
    return head + "\n" + block(encode(design(key), code))


def _questions(key, code):
    """بيرجع (أسئلة عربي، أسئلة إنجليزي) — أول سؤال هو الصيغة الأساسية"""
    fn, ar, fem, ar_alt, en, en_alt, default = OBJECTS[key]
    if key in FIXED_COLOR:
        names, en_names = [ar] + ar_alt, [en] + en_alt
    else:
        c = COLOR_BY_CODE[code]
        main = c[2] if fem else c[1]
        names = [f"{ar} {main}"] + [f"{a} {main}" for a in ar_alt]
        names += [f"{ar} {w}" for w in ({c[1], c[2], *c[3]} - {main})]
        names += [f"{ar} لونه {c[1]}", f"{ar} باللون ال{c[1]}"]
        en_names = [f"{c[4]} {en}"] + [f"{c[4]} {e}" for e in en_alt] + [f"{w} {en}" for w in c[5]]
        if code == default:                                # من غير لون = اللون الافتراضي
            names += [ar] + ar_alt
            en_names += [en] + en_alt
    return ([t.format(x=n) for n in names for t in AR_TEMPLATES],
            [t.format(x=n) for n in en_names for t in EN_TEMPLATES])


def combos(include_held_out=False):
    """كل (رسمة، لون) للتدريب، أو المتشالين للاختبار"""
    out = []
    for key, obj in OBJECTS.items():
        codes = [obj[6]] if key in FIXED_COLOR else [c[0] for c in COLORS]
        for code in codes:
            if ((key, code) in HELD_OUT) == include_held_out:
                out.append((key, code))
    return out


def build(include_held_out=False):
    """بيرجع قايمة (أسئلة، رد). بـ include_held_out=True بيرجع التركيبات المتشالة للاختبار بس."""
    out = []
    for key, code in combos(include_held_out):
        ar_qs, en_qs = _questions(key, code)
        out += [(ar_qs, answer(key, code, True)), (en_qs, answer(key, code, False))]
    return out


# ============================================================
# فهم الطلب: إيه الرسومات والألوان المطلوبة
# ============================================================
def _strip_al(word):
    return word[2:] if word.startswith("ال") and len(word) > 3 else word


def _norm(text):
    """نفس تنضيف الأسئلة (ئ → ي، ة → ه…) عشان الأسماء تطابق السؤال بعد normalize_question"""
    try:
        import os
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from tokenizer import normalize_question
        return normalize_question(text)
    except Exception:
        return text


def _names():
    names = []
    for key, (fn, ar, fem, ar_alt, en, en_alt, default) in OBJECTS.items():
        for name in [ar] + ar_alt + [en] + en_alt:
            names.append(([_strip_al(w) for w in _norm(name).split()], key))
    return sorted(names, key=lambda item: -len(item[0]))       # الاسم الأطول الأول


_NAME_LIST = _names()
_COLOR_WORDS = {}
for _c in COLORS:
    for _w in [_c[1], _c[2], _c[4]] + _c[3] + _c[5]:
        _COLOR_WORDS[_norm(_w)] = _c[0]
        _COLOR_WORDS["ال" + _norm(_w)] = _c[0]


_FIRST_WORDS = {w for name, _ in _NAME_LIST for w in name}       # كل كلمات أسماء الرسومات
_PREFIXES = ("وال", "بال", "لل", "ال", "و", "ب", "ل")


def _bare(word):
    """بيشيل حروف العطف والجر والتعريف من أول الكلمة لو اللي بعدها اسم رسمة أو لون
    ("والقمر" → "قمر"، "للشمس" → "شمس"، "وسودا" → "سودا")"""
    if word in _FIRST_WORDS or word in _COLOR_WORDS:
        return _strip_al(word) if word not in _FIRST_WORDS else word
    for p in _PREFIXES:
        rest = word[len(p):]
        if word.startswith(p) and len(rest) >= 2 and (rest in _FIRST_WORDS or rest in _COLOR_WORDS):
            return rest
    return word


def find_objects(question, limit=3):
    """
    بيدوّر في الطلب (بعد التنضيف) على الرسومات المعروفة وألوانها بالترتيب.
    بيرجع قايمة (المفتاح، رمز اللون أو None). "قطه سودا وشمس" → [(cat, K), (sun, None)]
    اللون بيتاخد من الكلمة اللي بعد الاسم (عربي) أو اللي قبله (إنجليزي).
    """
    words = [_bare(w) for w in question.split()]
    found, i, used = [], 0, set()
    while i < len(words):
        hit = None
        for name, key in _NAME_LIST:
            if words[i:i + len(name)] == name:
                hit = (key, len(name))
                break
        if hit is None:
            i += 1
            continue
        key, n = hit
        color = None
        for j in range(i + n, min(i + n + 3, len(words))):          # بعد الاسم: "قطه لونها اسود"
            if any(words[j:j + len(nm)] == nm for nm, _ in _NAME_LIST):
                break
            if words[j] in _COLOR_WORDS and j not in used and not words[j].isascii():
                color = _COLOR_WORDS[words[j]]
                used.add(j)
                break
        if (color is None and i > 0 and words[i - 1] in _COLOR_WORDS and (i - 1) not in used
                and words[i - 1].isascii()):
            color = _COLOR_WORDS[words[i - 1]]                       # قبل الاسم: "black cat"
            used.add(i - 1)
        if key in FIXED_COLOR:
            color = None
        found.append((key, color))
        i += n
        if len(found) >= limit:
            break
    return found


def what_was_asked(question):
    """أول رسمة مطلوبة: (المفتاح أو None، رمز اللون أو None)"""
    found = find_objects(question, limit=1)
    return found[0] if found else (None, None)


def known_names(arabic=True):
    return [obj[1] if arabic else obj[4] for obj in OBJECTS.values()]


# ============================================================
# معاينة
# ============================================================
def sheet(path, cell=8, per_row=9):
    """بيرسم كل الرسومات في صورة واحدة عشان نراجعها بالعين"""
    import os
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import vision
    keys = list(OBJECTS)
    n_rows = (len(keys) + per_row - 1) // per_row
    size = N * cell + cell
    img = np.full((n_rows * size + cell, per_row * size + cell, 4), 255, dtype=np.uint8)
    img[..., :3] = 60
    for i, key in enumerate(keys):
        oy, ox = (i // per_row) * size + cell, (i % per_row) * size + cell
        for y, row in enumerate(grid_for(key)):
            for x, ch in enumerate(row):
                rgb = tuple(int(PALETTE[ch][k:k + 2], 16) for k in (1, 3, 5))
                img[oy + y * cell:oy + (y + 1) * cell, ox + x * cell:ox + (x + 1) * cell, :3] = rgb
    with open(path, "wb") as f:
        f.write(vision.encode_png(img))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--sheet":
        sheet(sys.argv[2])
        print("اتحفظت:", sys.argv[2])
    else:
        lens = [len(encode(design(k), OBJECTS[k][6])) for k in OBJECTS]
        print(f"{len(OBJECTS)} رسمة | {len(combos())} تركيبة تدريب | {len(combos(True))} تركيبة اختبار")
        print(f"طول الرسمة المكتوبة: متوسط {sum(lens) / len(lens):.0f} حرف، أقصى {max(lens)}")
        print(answer("cat", "K"))
        print("\n".join(grid_for("cat", "K")))
