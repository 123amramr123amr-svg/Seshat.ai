"""
شات مع seshat.chet من التيرمنال (من غير متصفح)

بيستخدم المخ الكامل (brain.py): أدب + حساب + رسم + بحث النت + تعلم.

    python chat.py
    python chat.py "اكتب كود بايثون يجمع رقمين"
"""

import sys

from engine import Seshat
from brain import Brain


def main():
    seshat = Seshat()
    brain = Brain(seshat)
    if len(sys.argv) > 1:
        print(brain.reply(" ".join(sys.argv[1:]))["response"])
        return
    print(f"seshat.chet — {seshat.num_params():,} معامل ({seshat.engine_name}) "
          f"+ المخ (بحث/تعلم/صور). اكتب 'خروج' عشان تقفل.\n")
    while True:
        try:
            msg = input("إنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg in ("خروج", "exit", "quit"):
            break
        if not msg:
            continue
        r = brain.reply(msg)
        if r["thinking"]:
            print(f"  🧠 {r['thinking']}")
        if r.get("image"):
            print(f"  🖼 {r['image']}")
        print(f"seshat: {r['response']}\n      (ثقة {r['confidence']:.0%} — {r['seconds']} ث)\n")


if __name__ == "__main__":
    main()
