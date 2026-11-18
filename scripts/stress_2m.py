"""
خ12 — اختبار الضغط الكامل: ٢٠٠٠٠٠٠ حرف إدخال وتوليد طويل
بيشتغل على محرك torch (الأسرع في التدفق) على أي ملف نموذج.
"""
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "backups/seshat_k11_stable.npz"
TARGET = 2_000_000
OUT_TOKENS = int(os.environ.get("STRESS_OUT_TOKENS", "6000"))

print(f"النموذج: {MODEL}")
from engine import TorchSeshat

t0 = time.time()
s = TorchSeshat(MODEL)
print(f"تحميل المحرك: {time.time() - t0:.1f} ث | info: max_context={s.info()['max_context']}")

# ---------- بناء مستند ٢ مليون حرف ----------
t0 = time.time()
chunk = ("بند %d من العقد: التزام الطرفين بتسليم المستندات في موعدها المحدد، "
         "وعلى الطرف الأول توفير الدعم الفني طوال مدة الاتفاق بين الطرفين.\n")
doc_parts = []
n = 0
while sum(len(p) for p in doc_parts) < TARGET - 2000:
    doc_parts.append(chunk % (n + 1))
    n += 1
needle = "\nملاحظة مهمة في نهاية العقد: رقم التأمين النهائي هو 4712\n"
doc = "".join(doc_parts) + needle
doc = doc[:TARGET] if len(doc) > TARGET else doc
print(f"المستند: {len(doc):,} حرف (بني في {time.time() - t0:.1f} ث)")

# ---------- التدفق الكامل + السؤال ----------
q = "ايه رقم التأمين النهائي اللي في آخر العقد"
print(f"\n--- تدفق {len(doc):,} حرف + سؤال (محرك torch) ---")
t0 = time.time()
r = s.reply_long(doc, q, max_new_tokens=120)
dt = time.time() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
rate = len(doc) / dt / 1000
print(f"الرد: {r['response'][:120]!r}")
print(f"سطر التفكير: {r['thinking'][:80]!r}")
print(f"الوقت: {dt:.1f} ث | السرعة: {rate:.0f} ألف حرف/ث | ذروة الرام: {peak_mb:.0f} MB")
print(f"حروف المستند المعالجة: {r['doc_chars']:,} | توكنز البرومبت: {r['prompt_tokens']:,}")

# ---------- توليد طويل (من غير قفل block_size) ----------
print(f"\n--- توليد {OUT_TOKENS} توكن من غير قفل block_size ---")
prompt = s.tok.encode_prompt("احكي قصة قصيرة عن البحر")
t0 = time.time()
ids, probs = s._generate_long(prompt, OUT_TOKENS, 0.7)
dt = time.time() - t0
text = s.tok.decode(ids)
print(f"متولد: {len(ids)} توكن ({len(text)} حرف) في {dt:.1f} ث "
      f"({len(ids) / dt:.0f} توكن/ث) | متوسط الثقة: {sum(probs) / max(1, len(probs)):.3f}")
print(f"أول 200 حرف من المولد: {text[:200]!r}")

print("\n✅ اختبار الضغط خلص")
