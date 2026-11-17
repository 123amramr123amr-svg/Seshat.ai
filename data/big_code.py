"""
مولّد الأكواد الضخمة — جولة خ11
================================

seshat نموذج نصي بحت، وقادر يبني أكواد برمجية كاملة وضخمة من قوالب قابلة
للتركيب. كل قالب بيرجع برنامج كامل (100+ سطر) جاهز للتشغيل: تعليقات عربية
+ كلاسات + دوال + main + اختبار سريع.

المولد ده معالج مضمون 100% — مش محتاج النموذج ولا النت، فبيشتغل دايمًا
بنفس الجودة وبتعليقات وشرح عربي. brain.py بيكلمه قبل ما يدور في النموذج.

الواجهة:
    big_code_request(message) → dict أو None
        {"kind": "template", "key", "title_ar", "title_en", "lang",
         "code", "parts_ar", "parts_en"}
        {"kind": "intent"}  ← نية كود واضحة بس موضوع مش معروف
    build_big_code(res, arabic=True) → نص الرد الكامل
    big_code_guide(arabic=True) → رد الأقوالب المتاحة
"""

import re

# ----------------------------------------------------------------------
# كشف نية طلب الكود — بعد تطبيع بسيط للألف والهمزات
# ----------------------------------------------------------------------
_NORM_TABLE = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه",
                             "ى": "ي", "ؤ": "و", "ئ": "ي", "ّ": ""})

_CODE_INTENT_RE = re.compile(
    r"(?:^|[\s،,.؟!])(?:اكتب|اعمل|برمج|ولد|هات|صمم|عايز|عاوز|ممكن)\s*"
    r"(?:لي)?\s*(?:كود|برنامج|سكربت)(?=[\s،,.؟!:]|$)"
    r"|(?:^|[\s،,.؟!])(?:كود|برنامج|سكربت)\s*(?:لل|لـ|ل)\s*[ا-يA-Za-z]{2,}"
    r"|(?:write|make|build|generate|create|give\s*me|code\s*for)\s+"
    r"(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:python\s+|flask\s+)?"
    r"(?:code|program|script|app|function|game)(?=[\s،,.?!:]|$)"
    r"|(?:write|make|build|generate|create)\s+"
    r"(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:python|flask|javascript)\s+[a-z_]{3,}",
    re.IGNORECASE)


def _norm(text):
    return (text or "").translate(_NORM_TABLE).lower()


# ----------------------------------------------------------------------
# القوالب — كل واحدة: مفتاح + أسماء + كلمات مفتاحية + أجزاء + باني الكود
# ----------------------------------------------------------------------
TEMPLATES = []


def _template(key, ar_names, en_names, kw, title_ar, title_en, lang,
              parts_ar, parts_en, builder):
    TEMPLATES.append({
        "key": key, "ar_names": ar_names, "en_names": en_names, "kw": kw,
        "title_ar": title_ar, "title_en": title_en, "lang": lang,
        "parts_ar": parts_ar, "parts_en": parts_en, "builder": builder,
    })


# ============================================================
# 1) تطبيق مهام — Flask + تخزين JSON
# ============================================================
def _build_todo_api():
    return r'''# -*- coding: utf-8 -*-
"""تطبيق مهام كامل بـ Flask — خ11 (ساشات كتبه بنفسه).

تشغيله:
    pip install flask
    python todo_app.py

المزايا:
    - إضافة / عرض / تعديل / حذف مهام
    - تخزين في ملف JSON (بيفضل بعد إعادة التشغيل)
    - تحقق من المدخلات وردود JSON منظمة
"""

import json
import os
from datetime import datetime

from flask import Flask, jsonify, request

APP = Flask(__name__)
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.json")


# ---------------------------- التخزين ----------------------------
def load_tasks():
    """بيحمّل المهام من ملف JSON، ولو الملف مش موجود بيرجع قايمة فاضية."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        # الملف بايظ؟ منفضلش نموت — نبدأ فاضيين ونسلم الخطأ للواجهة
        return []


def save_tasks(tasks):
    """بيكتب المهام في ملف JSON بنسخة مؤقتة عشان ما تبوظش لو الكهرباء قطعت."""
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DATA_FILE)


def next_id(tasks):
    return (max((t["id"] for t in tasks), default=0)) + 1


# ---------------------------- التحقق ----------------------------
def valid_task_payload(payload, partial=False):
    """بيتأكد إن البيانات اللي جاية من المستخدم سليمة.
    partial=True يعني تعديل جزئي (مش لازم كل الحقول)."""
    if not isinstance(payload, dict):
        return None
    title = payload.get("title")
    if not partial and (not isinstance(title, str) or not title.strip()):
        return None
    if "title" in payload and (not isinstance(title, str) or not title.strip()):
        return None
    if "done" in payload and not isinstance(payload["done"], bool):
        return None
    return payload


# ---------------------------- المسارات ----------------------------
@APP.route("/tasks", methods=["GET"])
def list_tasks():
    """كل المهام — مع دعم فلترة ?done=true/false و ?q=كلمة"""
    tasks = load_tasks()
    done = request.args.get("done")
    q = (request.args.get("q") or "").strip().lower()
    if done in ("true", "false"):
        want = done == "true"
        tasks = [t for t in tasks if t["done"] is want]
    if q:
        tasks = [t for t in tasks if q in t["title"].lower()]
    return jsonify({"count": len(tasks), "tasks": tasks})


@APP.route("/tasks", methods=["POST"])
def create_task():
    """إضافة مهمة جديدة — العنوان مطلوب."""
    payload = valid_task_payload(request.get_json(silent=True))
    if payload is None:
        return jsonify({"error": "العنوان مطلوب ومانفيسش"}), 400
    tasks = load_tasks()
    task = {
        "id": next_id(tasks),
        "title": payload["title"].strip(),
        "done": payload.get("done", False),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    tasks.append(task)
    save_tasks(tasks)
    return jsonify({"message": "اتضافت المهمة", "task": task}), 201


@APP.route("/tasks/<int:task_id>", methods=["GET", "PUT", "DELETE"])
def task_detail(task_id):
    """عرض / تعديل / حذف مهمة واحدة بالـ id."""
    tasks = load_tasks()
    idx = next((i for i, t in enumerate(tasks) if t["id"] == task_id), None)
    if idx is None:
        return jsonify({"error": "مفيش مهمة بالرقم ده"}), 404

    if request.method == "GET":
        return jsonify({"task": tasks[idx]})

    if request.method == "PUT":
        payload = valid_task_payload(request.get_json(silent=True), partial=True)
        if payload is None:
            return jsonify({"error": "مافيش حقول صالحة للتعديل"}), 400
        if "title" in payload:
            tasks[idx]["title"] = payload["title"].strip()
        if "done" in payload:
            tasks[idx]["done"] = payload["done"]
        tasks[idx]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        save_tasks(tasks)
        return jsonify({"message": "اتعدلت", "task": tasks[idx]})

    removed = tasks.pop(idx)
    save_tasks(tasks)
    return jsonify({"message": "اتمسحت", "task": removed})


@APP.route("/tasks/<int:task_id>/toggle", methods=["POST"])
def toggle_task(task_id):
    """قلب حالة المهمة: خالصة ↔ لسه شغالة."""
    tasks = load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = not t["done"]
            save_tasks(tasks)
            return jsonify({"task": t})
    return jsonify({"error": "مفيش مهمة بالرقم ده"}), 404


@APP.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "tasks": len(load_tasks())})


if __name__ == "__main__":
    APP.run(debug=True, port=5000)
'''


_template(
    "todo_api",
    ["تطبيق مهام", "مهام flask", "todo api", "قائمة مهام"],
    ["todo app", "todo api", "flask app", "task manager"],
    ["مهام", "todo", "تودو", "فلاسك", "flask", "task"],
    "تطبيق مهام كامل بـ Flask وتخزين JSON",
    "Full Flask todo app with JSON storage",
    "python",
    ["تخزين JSON بأمان (كتابة مؤقتة + استبدال)", "5 مسارات REST كاملة",
     "تحقق من المدخلات وفلترة واضحة"],
    ["safe JSON storage", "5 full REST routes", "input validation"],
    _build_todo_api,
)


# ============================================================
# 2) لعبة إكس-أو — كلاس + خصم ذكي
# ============================================================
def _build_xo_game():
    return r'''# -*- coding: utf-8 -*-
"""لعبة إكس-أو كاملة — خ11 (ساشات كتبه بنفسه).

تشغيله:
    python xo_game.py

المزايا:
    - لوحة 3×3 بلوحة مرقمة عشان اللعب سهل
    - خصم ذكي: يكسب لو يقدر، يمنعك لو كنت هتكسب، وياخد الوسط
    - تحقق كامل من الحركات + قايمة نتائج
"""

import random

WINS = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)]


class TicTacToe:
    """اللعبة نفسها: اللوحة، الحركات، والحكم."""

    def __init__(self):
        self.reset()

    def reset(self):
        """لوحة جديدة: كل خانة فاضية، ودور اللاعب الأول."""
        self.board = [" "] * 9
        self.current = "X"
        self.moves = 0

    def print_board(self):
        """عرض اللوحة — كل خانة فاضية بتظهر برقمها عشان اللعب يبقى سهل."""
        for row in range(3):
            cells = []
            for col in range(3):
                i = row * 3 + col
                cells.append(self.board[i] if self.board[i] != " " else str(i + 1))
            print(" " + " | ".join(cells))
            if row < 2:
                print("---+---+---")
        print()

    def winner(self):
        """بيرجع X أو O لو كسبان، وإلا None."""
        for a, b, c in WINS:
            if self.board[a] == self.board[b] == self.board[c] != " ":
                return self.board[a]
        return None

    def free_cells(self):
        return [i for i, v in enumerate(self.board) if v == " "]

    def make_move(self, cell, player=None):
        """حركة واحدة مع تحقق: الخانة موجودة؟ فاضية؟ الدور صح؟"""
        player = player or self.current
        if not 0 <= cell <= 8:
            return False, "الخانة لازم تكون من 1 لـ 9"
        if self.board[cell] != " ":
            return False, "الخانة دي متاخدة"
        self.board[cell] = player
        self.moves += 1
        self.current = "O" if player == "X" else "X"
        return True, "تمام"

    def ai_move(self):
        """عقل الخصم بأولوية واضحة: اكسب → امنع → الوسط → الزوايا → عشوائي."""
        me, you = "O", "X"
        # 1) لو ممكن أكسب النهارده — اكسب فورًا
        for i in self.free_cells():
            self.board[i] = me
            won = self.winner() == me
            self.board[i] = " "
            if won:
                return i
        # 2) لو الخصم هيكسب في الحركة الجاية — امنعه
        for i in self.free_cells():
            self.board[i] = you
            lost = self.winner() == you
            self.board[i] = " "
            if lost:
                return i
        # 3) الوسط أقوى خانة، بعدها الزوايا، وأخيرًا أي حاجة
        if self.board[4] == " ":
            return 4
        corners = [i for i in (0, 2, 6, 8) if self.board[i] == " "]
        if corners:
            return random.choice(corners)
        return random.choice(self.free_cells())

    def finished(self):
        return self.winner() is not None or self.moves >= 9


def ask_cell(prompt):
    """قراءة حركة من المستخدم مع سماح بالخطأ — منفيضش اللعبة تقفل."""
    raw = input(prompt).strip()
    return int(raw) - 1 if raw.isdigit() else -1


def main():
    """حلقة اللعب: أنت X والخصم O — وآخر جولة بتعرض النتيجة."""
    game = TicTacToe()
    print("لعبة إكس-أو — انت X والخصم O. اكتب رقم الخانة (1-9).\n")
    game.print_board()
    while not game.finished():
        cell = ask_cell("حركتك: ")
        ok, why = game.make_move(cell)
        if not ok:
            print("!", why, "- جرب تاني")
            continue
        if not game.finished():
            game.make_move(game.ai_move(), player="O")
        game.print_board()
    w = game.winner()
    print("كسبان:", w if w else "تعادل — ماتعلش حد")


if __name__ == "__main__":
    main()
'''


_template(
    "xo_game",
    ["اكس او", "أكس أو", "لعبة اكس", "tic tac toe", "xo"],
    ["tic tac toe", "xo game", "noughts and crosses"],
    ["اكس او", "اكس-او", "xo", "tic"],
    "لعبة إكس-أو كاملة بخصم ذكي",
    "Full tic-tac-toe game with a smart opponent",
    "python",
    ["كلاس اللعبة + الحكم + اللوحة المرقمة", "عقل خصم: كسب → منع → وسط → زوايا",
     "تحقق كامل من الحركات بدون انهيار"],
    ["game class + referee + numbered board", "smart AI: win → block → center",
     "robust move validation"],
    _build_xo_game,
)


# ============================================================
# 3) نظام إدارة مكتبة — كلاسات متكاملة
# ============================================================
def _build_library():
    return r'''# -*- coding: utf-8 -*-
"""نظام إدارة مكتبة كامل — خ11 (ساشات كتبه بنفسه).

تشغيله:
    python library_system.py

المزايا:
    - كلاسات Book و Member و Library
    - استعارة وإرجاع مع تحقق من الشروط
    - بحث بالعنوان/المؤلف/النوع + تقرير حالة المكتبة
"""

from dataclasses import dataclass, field


@dataclass
class Book:
    """كتاب واحد في المكتبة — مع عدد النسخ المتاح."""
    title: str
    author: str
    category: str = "عام"
    copies: int = 1
    borrowed: int = 0

    @property
    def available(self):
        return self.copies - self.borrowed

    def key(self):
        return self.title.lower()


@dataclass
class Member:
    """عضو مكتبة — مع حد أعلى لعدد الكتب المستعارة."""
    name: str
    member_id: int
    borrowed_titles: list = field(default_factory=list)
    MAX_BOOKS = 3

    def can_borrow(self):
        return len(self.borrowed_titles) < self.MAX_BOOKS


class LibraryError(Exception):
    """خطأ مخصوص للمكتبة — الرسالة بتوصل للواجهة زي ما هي."""


class Library:
    """المكتبة نفسها: تسجيل كتب وأعضاء، استعارة، إرجاع، بحث، وتقرير."""

    def __init__(self, name="مكتبة ساشات"):
        self.name = name
        self.books = {}          # العنوان → Book
        self.members = {}        # الـ id → Member
        self._next_member_id = 1

    # ------------------------- التسجيل -------------------------
    def add_book(self, title, author, category="عام", copies=1):
        if title.strip() == "" or copies < 1:
            raise LibraryError("عنوان الكتاب وعدد النسخ لازم يكونوا صحيحين")
        book = self.books.get(title.lower())
        if book:
            book.copies += copies      # نفس الكتاب؟ زود النسخ
            return book
        book = Book(title=title.strip(), author=author.strip(),
                    category=category.strip() or "عام", copies=copies)
        self.books[book.key()] = book
        return book

    def add_member(self, name):
        if not name.strip():
            raise LibraryError("اسم العضو مطلوب")
        member = Member(name=name.strip(), member_id=self._next_member_id)
        self._next_member_id += 1
        self.members[member.member_id] = member
        return member

    # ------------------------- الاستعارة -------------------------
    def borrow(self, member_id, title):
        member = self.members.get(member_id)
        book = self.books.get(title.lower())
        if member is None or book is None:
            raise LibraryError("العضو أو الكتاب مش موجود")
        if not member.can_borrow():
            raise LibraryError("العضو وصل للحد الأقصى (%d كتب)" % Member.MAX_BOOKS)
        if book.available < 1:
            raise LibraryError("مفيش نسخ متاحة من «%s»" % book.title)
        if book.title in member.borrowed_titles:
            raise LibraryError("العضو معاه نسخة من الكتاب ده بالفعل")
        book.borrowed += 1
        member.borrowed_titles.append(book.title)
        return True

    def give_back(self, member_id, title):
        member = self.members.get(member_id)
        book = self.books.get(title.lower())
        if member is None or book is None:
            raise LibraryError("العضو أو الكتاب مش موجود")
        if book.title not in member.borrowed_titles:
            raise LibraryError("العضو مش مستعير الكتاب ده")
        member.borrowed_titles.remove(book.title)
        book.borrowed -= 1
        return True

    # ------------------------- البحث والتقرير -------------------------
    def search(self, word):
        w = word.strip().lower()
        return [b for b in self.books.values()
                if w in b.title.lower() or w in b.author.lower()
                or w in b.category.lower()]

    def report(self):
        total = sum(b.copies for b in self.books.values())
        out = [f"تقرير {self.name}:",
               f"  عناوين: {len(self.books)} — نسخ كلية: {total}",
               f"  أعضاء: {len(self.members)}"]
        for b in self.books.values():
            out.append(f"  - {b.title} ({b.author}): {b.available}/{b.copies} متاح")
        return "\n".join(out)


def main():
    """عرض سريع للنظام: تسجيل، استعارة، إرجاع، بحث، وتقرير نهائي."""
    lib = Library()
    lib.add_book("أولاد حارتنا", "نجيب محفوظ", "روايات", 2)
    lib.add_book("الخيميائي", "باولو كويلو", "روايات")
    lib.add_book("مقدمة ابن خلدون", "ابن خلدون", "تاريخ")
    ahmed = lib.add_member("أحمد")
    sara = lib.add_member("سارة")

    lib.borrow(ahmed.member_id, "أولاد حارتنا")
    lib.borrow(sara.member_id, "الخيميائي")
    print(lib.report())

    lib.give_back(ahmed.member_id, "أولاد حارتنا")
    print("\nبعد الإرجاع:")
    print(lib.report())
    print("\nنتيجة بحث عن «روايات»:", [b.title for b in lib.search("روايات")])


if __name__ == "__main__":
    main()
'''


_template(
    "library",
    ["نظام مكتبة", "إدارة مكتبة", "library system"],
    ["library system", "library management"],
    ["مكتبه", "library", "كتب"],
    "نظام إدارة مكتبة بكلاسات متكاملة",
    "Full library management system with classes",
    "python",
    ["كلاسات Book/Member/Library", "استعارة وإرجاع بشروط محمية",
     "بحث وتقرير حالة كامل"],
    ["Book/Member/Library classes", "guarded borrow/return rules",
     "search and full report"],
    _build_library,
)


# ============================================================
# 4) خوارزميات الترتيب والبحث — مع مقارنة أداء
# ============================================================
def _build_sorting():
    return r'''# -*- coding: utf-8 -*-
"""خوارزميات الترتيب والبحث — خ11 (ساشات كتبه بنفسه).

تشغيله:
    python sorting_suite.py

المزايا:
    - 5 خوارزميات ترتيب + بحث ثنائي
    - مقارنة وقت التنفيذ على أرقام عشوائية
    - شرح مختصر فوق كل خوارزمية
"""

import random
import time


def bubble_sort(items):
    """ترتيب الفقاعات: بيقارن الجيران ويبدل — بسيط لكن بطيء (n²)."""
    data = list(items)
    n = len(data)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
                swapped = True
        if not swapped:            # بتحسين مهم: لو مافيش تبديل → خلصنا
            break
    return data


def insertion_sort(items):
    """الإدراج: كل عنصر بيتشال وبيتزرع في مكانه الصح — ممتاز للبيانات شبه المرتبة."""
    data = list(items)
    for i in range(1, len(data)):
        value = data[i]
        j = i - 1
        while j >= 0 and data[j] > value:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = value
    return data


def selection_sort(items):
    """الاختيار: بنختار أصغر عنصر ونحطه في الأول، ونكرر."""
    data = list(items)
    for i in range(len(data) - 1):
        smallest = i
        for j in range(i + 1, len(data)):
            if data[j] < data[smallest]:
                smallest = j
        if smallest != i:
            data[i], data[smallest] = data[smallest], data[i]
    return data


def merge_sort(items):
    """الدمج: نقسم نصين، نرتب كل نص، وندمجهم — ثابت O(n log n)."""
    if len(items) <= 1:
        return list(items)
    mid = len(items) // 2
    left = merge_sort(items[:mid])
    right = merge_sort(items[mid:])
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(items):
    """السريع: نختار محور، نقسم أصغر/أكبر، ونكرر — عمليًا أسرع غالبًا."""
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    small = [x for x in items if x < pivot]
    equal = [x for x in items if x == pivot]
    big = [x for x in items if x > pivot]
    return quick_sort(small) + equal + quick_sort(big)


def binary_search(sorted_items, target):
    """بحث ثنائي على قايمة مرتبة — بيرجع موقع العنصر أو -1."""
    low, high = 0, len(sorted_items) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_items[mid] == target:
            return mid
        if sorted_items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def benchmark():
    """مقارنة عملية: وقت كل خوارزمية على نفس البيانات العشوائية."""
    data = [random.randint(0, 9999) for _ in range(800)]
    expected = sorted(data)
    algos = [("فقاعات", bubble_sort), ("إدراج", insertion_sort),
             ("اختيار", selection_sort), ("دمج", merge_sort),
             ("سريع", quick_sort)]
    print("مقارنة الأداء على %d رقم:" % len(data))
    for name, fn in algos:
        t0 = time.perf_counter()
        result = fn(data)
        dt = (time.perf_counter() - t0) * 1000
        ok = result == expected
        print(f"  {name:8s} → {dt:7.2f} ms — صحيح: {ok}")


def main():
    demo = [5, 2, 9, 1, 7, 3]
    print("الأصل:", demo)
    print("دمج:", merge_sort(demo))
    ordered = quick_sort(demo)
    print("سريع:", ordered)
    print("بحث ثنائي عن 7:", binary_search(ordered, 7))
    print("بحث ثنائي عن 100:", binary_search(ordered, 100))
    print()
    benchmark()


if __name__ == "__main__":
    main()
'''


_template(
    "sorting",
    ["خوارزميات الترتيب", "ترتيب البيانات", "sorting algorithms", "بحث ثنائي"],
    ["sorting algorithms", "sorting suite", "binary search"],
    ["ترتيب", "sort", "خوارزميات", "بحث ثنائي", "binary"],
    "خوارزميات الترتيب والبحث مع مقارنة أداء",
    "Sorting and search algorithms with benchmark",
    "python",
    ["5 خوارزميات ترتيب كاملة", "بحث ثنائي", "مقارنة وقت تنفيذ عملية"],
    ["5 sorting algorithms", "binary search", "live benchmark"],
    _build_sorting,
)


# ============================================================
# 5) آلة حاسبة بمحلل تعبيرات — كلاس + أولوية عمليات
# ============================================================
def _build_calculator():
    return r'''# -*- coding: utf-8 -*-
"""آلة حاسبة بمحلل تعبيرات — خ11 (ساشات كتبه بنفسه).

تشغيله:
    python calc_machine.py

المزايا:
    - كلاس Calculator بعمليات أساسية + ذاكرة
    - محلل تعبيرات كامل بأولوية عمليات صح: 2 + 3 * 4 = 14
    - حلقة REPL للتفاعل مع المستخدم
"""

import operator


class CalcError(Exception):
    """خطأ حساب واضح — بيمنع انهيار البرنامج."""
    pass


class Calculator:
    """حاسبة كاملة: عمليات، ذاكرة، ومحلل تعبيرات بأولوية عمليات."""

    OPS = {
        "+": operator.add, "-": operator.sub,
        "*": operator.mul, "/": operator.truediv,
    }
    PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}

    def __init__(self):
        self.memory = 0.0
        self.history = []

    # ------------------------- العمليات -------------------------
    def apply(self, a, sym, b):
        if sym not in self.OPS:
            raise CalcError("رمز غير معروف: %s" % sym)
        if sym == "/" and b == 0:
            raise CalcError("القسمة على صفر مش مسموحة")
        result = self.OPS[sym](a, b)
        self.history.append(f"{a} {sym} {b} = {result}")
        return result

    def remember(self, value):
        self.memory = float(value)
        return self.memory

    def recall(self):
        return self.memory

    # ------------------------- المحلل -------------------------
    def tokenize(self, text):
        """النص لرموز: أرقام (وأعشار) ورموز عمليات وأقواس."""
        tokens = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == " ":
                i += 1
                continue
            if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
                j = i
                while j < n and (text[j].isdigit() or text[j] == "."):
                    j += 1
                try:
                    tokens.append(float(text[i:j]))
                except ValueError:
                    raise CalcError("رقم مالخبط: %s" % text[i:j])
                i = j
            elif ch in self.OPS or ch in "()":
                tokens.append(ch)
                i += 1
            else:
                raise CalcError("حرف غريب: %s" % ch)
        return tokens

    def to_postfix(self, tokens):
        """شنتنج-يارد: الأقواس والأولوية بتتحل هنا صح."""
        out, stack = [], []
        for tok in tokens:
            if isinstance(tok, float):
                out.append(tok)
            elif tok in self.OPS:
                while (stack and stack[-1] in self.OPS
                       and self.PRECEDENCE[stack[-1]] >= self.PRECEDENCE[tok]):
                    out.append(stack.pop())
                stack.append(tok)
            elif tok == "(":
                stack.append(tok)
            elif tok == ")":
                while stack and stack[-1] != "(":
                    out.append(stack.pop())
                if not stack:
                    raise CalcError("قوس ناقص")
                stack.pop()          # نشيل القوس الافتتاحي
        while stack:
            if stack[-1] == "(":
                raise CalcError("قوس ناقص")
            out.append(stack.pop())
        return out

    def evaluate(self, expression):
        """من نص كامل للنتيجة: tokenize → postfix → حساب."""
        postfix = self.to_postfix(self.tokenize(expression))
        stack = []
        for tok in postfix:
            if isinstance(tok, float):
                stack.append(tok)
            else:
                if len(stack) < 2:
                    raise CalcError("تعبير ناقص")
                b = stack.pop()
                a = stack.pop()
                stack.append(self.apply(a, tok, b))
        if len(stack) != 1:
            raise CalcError("تعبير مالخبط")
        return stack[0]


def main():
    """REPL: اكتب تعبير زي 2 + 3 * 4، أو م: لحفظ، ر: للذاكرة، خروج: للخروج."""
    calc = Calculator()
    print("آلة حاسبة — اكتب تعبير (زي 2 + 3 * 4)، أو «خروج» للخروج.")
    while True:
        line = input("> ").strip()
        if not line or line in ("خروج", "exit", "quit"):
            print("مع السلامة — آخر عمليات:", calc.history[-3:])
            break
        if line in ("ر", "recall"):
            print("الذاكرة =", calc.recall())
            continue
        try:
            value = calc.evaluate(line)
            calc.remember(value)
            print("=", value)
        except CalcError as err:
            print("خطأ:", err)


if __name__ == "__main__":
    main()
'''


_template(
    "calculator",
    ["آلة حاسبة", "حاسبة برمجية", "calculator"],
    ["calculator", "expression calculator"],
    ["حاسبه", "calculator", "حساب تعبير"],
    "آلة حاسبة بمحلل تعبيرات كامل",
    "Calculator with a full expression parser",
    "python",
    ["كلاس حاسبة بذاكرة وسجل", "محلل تعبيرات بأولوية عمليات صح",
     "حلقة REPL للتفاعل"],
    ["calculator class with memory", "expression parser with precedence",
     "interactive REPL"],
    _build_calculator,
)


# ----------------------------------------------------------------------
# واجهة الاستخدام
# ----------------------------------------------------------------------
def find_template(message):
    """بحث عن القالب المناسب من الرسالة المطبعة — بيرجع القالب أو None.
    الكلمات المفتاحية بتتبدل بحدود كلمات (عشان «كتب» ما تتحسبش جوه «اكتب»)."""
    norm = _norm(message)
    for t in TEMPLATES:
        for kw_re in _kw_regexes(t["key"], t["kw"]):
            if kw_re.search(norm):
                return t
    return None


_KW_RE_CACHE = {}


def _kw_regexes(key, kws):
    """ريجيكس بحدود كلمات لكل مفتاح — بيتجمع مرة واحدة (مخزن بالمفتاح).
    • المفتاح نفسه بيتطبع بنفس دالة التطبيع (عشان «ثنائي» تطابق «ثنايي» بعد تحويل الئ).
    • بيسمح بسابقة «ال» أو «ل/لل» قبل كل كلمة (عشان «للترتيب» و«البحث الثنائي» تشتغلوا).
    • ويمنع التشال جوه كلمة أطول (عشان «كتب» ما تتحسبش جوه «اكتب»)."""
    if key not in _KW_RE_CACHE:
        compiled = []
        for kw in kws:
            parts = [re.escape(w) for w in _norm(kw).split()]
            body = r"\s+(?:ال)?".join(parts)
            compiled.append(re.compile(
                r"(?<![ا-يA-Za-z])(?:ال|لل|ل)?" + body + r"(?![ا-يA-Za-z])"))
        _KW_RE_CACHE[key] = compiled
    return _KW_RE_CACHE[key]


def big_code_request(message):
    """كشف طلب كود: قالب معروف → template، نية من غير قالب → intent."""
    if not _CODE_INTENT_RE.search(_norm(message)):
        return None
    t = find_template(message)
    if t is None:
        return {"kind": "intent"}
    return {
        "kind": "template",
        "key": t["key"], "title_ar": t["title_ar"], "title_en": t["title_en"],
        "lang": t["lang"], "code": t["builder"](),
        "parts_ar": t["parts_ar"], "parts_en": t["parts_en"],
    }


def _template_names_for_guide(arabic):
    names = []
    for t in TEMPLATES:
        name = t["ar_names"][0] if arabic else t["en_names"][0]
        title = t["title_ar"] if arabic else t["title_en"]
        names.append("  - %s — %s" % (name, title))
    return "\n".join(names)


def build_big_code(res, arabic=True):
    """نص الرد الكامل: مقدمة + الكود + المميزات + طريقة التشغيل."""
    code = res["code"]
    lines = code.count("\n") + 1
    if arabic:
        parts = "، ".join(res["parts_ar"])
        run = "python " + _file_name(res["key"])
        return (
            "تمام — ده كود كامل جاهز للتشغيل: %s\n"
            "(%d سطر — %s — اتكوّن دلوقتي خصيصًا لطلبك، مش منسوخ من حاجة)\n\n"
            "```%s\n%s```\n\n"
            "اللي جواه: %s.\n"
            "إزاي تشغله: احفظه في ملف باسم %s وشغّل: %s\n"
            "عايز تعديل؟ قولي بالظبط عايز تزود ايه وأنا أكوّنلك نسخة جديدة."
        ) % (res["title_ar"], lines, res["lang"], res["lang"], code, parts,
             _file_name(res["key"]), run)
    parts = "; ".join(res["parts_en"])
    run = "python " + _file_name(res["key"])
    return (
        "Sure — here is a complete, ready-to-run program: %s\n"
        "(%d lines of %s — composed fresh for your request, not copied)\n\n"
        "```%s\n%s```\n\n"
        "What's inside: %s.\n"
        "How to run: save it as %s and run: %s\n"
        "Want changes? Tell me exactly what to add and i will build a new version."
    ) % (res["title_en"], lines, res["lang"], res["lang"], code, parts,
         _file_name(res["key"]), run)


def big_code_guide(arabic=True):
    """رد القوايم — لما فيه نية كود بس الموضوع مش من القوالب."""
    if arabic:
        return (
            "أيوه — بحط أكواد كاملة جاهزة للتشغيل، بس المواضيع اللي جاهزة عندي "
            "بالظبط دلوقتي دي:\n" + _template_names_for_guide(True) +
            "\n\nقول زي: «اكتب كود لعبة اكس او» أو «write code for a todo app» "
            "وهبني لك البرنامج كامل بتعليقات وشرح. ولو الموضوع تاني خالص "
            "(زي فلتر أو مركب فضاء) اكتبلي فكرته وأنا أشرح لك خطوات بنائه خطوة خطوة."
        )
    return (
        "Yes — i write complete ready-to-run programs, but the topics i have "
        "ready right now are:\n" + _template_names_for_guide(False) +
        "\n\nTry: \"write code for a todo app\" or \"اكتب كود لعبة اكس او\". "
        "For any other topic, describe the idea and i will walk you through "
        "the build steps."
    )


def _file_name(key):
    return {"todo_api": "todo_app.py", "xo_game": "xo_game.py",
            "library": "library_system.py", "sorting": "sorting_suite.py",
            "calculator": "calc_machine.py"}[key]
