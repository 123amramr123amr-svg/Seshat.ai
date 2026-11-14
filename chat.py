"""
شات مع seshat.chet من التيرمنال (من غير متصفح)

    python chat.py
    python chat.py "اكتب كود بايثون يجمع رقمين"
"""

import sys

from engine import Seshat


def main():
    seshat = Seshat()
    if len(sys.argv) > 1:
        print(seshat.reply(" ".join(sys.argv[1:]))["response"])
        return
    print(f"seshat.chet — {seshat.num_params():,} معامل ({seshat.engine_name}). اكتب 'خروج' عشان تقفل.\n")
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
        r = seshat.reply(msg)
        if r["thinking"]:
            print(f"  🧠 {r['thinking']}")
        print(f"seshat: {r['response']}\n      (ثقة {r['confidence']:.0%} — {r['seconds']} ث)\n")


if __name__ == "__main__":
    main()
