"""
سيرفر Seshat.ai — واجهة Flask.

التشغيل:
    python app.py
ثم افتح http://localhost:5000

إعدادات النشر:
    SESHAT_ADMIN_TOKEN=<long-random-secret>  # مطلوب لتعديل /config
    SESHAT_CHAT_RATE_LIMIT=30                # طلبًا لكل IP خلال 60 ثانية
    SESHAT_MAX_MESSAGE_CHARS=8000            # أقصى طول للرسالة (السياق الاتسع)
"""

import hmac
import os
import threading
import time
from collections import deque

from flask import Flask, jsonify, render_template, request, send_file

from engine import Seshat, drawings, _trim_memory
from brain import Brain
from models import list_models

app = Flask(__name__)
app.json.ensure_ascii = False
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

# محدد معدل بسيط داخل العملية. في النشر متعدد العمال استخدم محددًا مشتركًا
# (مثل Redis) أمام التطبيق، ولا تعتمد على هذا المحدد وحده.
_RATE_WINDOW_SECONDS = 60
_rate_lock = threading.Lock()
_chat_requests = {}


def _chat_rate_limit():
    """يسمح بعدد محدود من طلبات الشات لكل عنوان IP في نافذة زمنية متحركة."""
    try:
        limit = max(1, min(300, int(os.environ.get("SESHAT_CHAT_RATE_LIMIT", "30"))))
    except ValueError:
        limit = 30
    now = time.monotonic()
    client = request.remote_addr or "unknown"
    with _rate_lock:
        # تنظيف دوري للمدخلات القديمة والحد من نمو الذاكرة.
        stale_before = now - _RATE_WINDOW_SECONDS
        for ip in list(_chat_requests):
            q = _chat_requests[ip]
            while q and q[0] <= stale_before:
                q.popleft()
            if not q:
                del _chat_requests[ip]
        if client not in _chat_requests and len(_chat_requests) >= 10000:
            return False
        q = _chat_requests.setdefault(client, deque())
        if len(q) >= limit:
            return False
        q.append(now)
    return True


def _authorized_config_write():
    """لا تسمح بتغيير إعدادات المخ إلا بتوكن إداري مضبوط في بيئة التشغيل."""
    expected = os.environ.get("SESHAT_ADMIN_TOKEN", "")
    supplied = request.headers.get("Authorization", "")
    if not expected:
        return False
    prefix = "Bearer "
    if not supplied.startswith(prefix):
        return False
    return hmac.compare_digest(supplied[len(prefix):], expected)


print("⏳ جاري تحميل النموذج...")
seshat = Seshat()
print(f"✅ اتحمل النموذج — {seshat.num_params():,} معامل | المحرك: {seshat.engine_name}")

print("⏳ تجهيز المخ...")
brain = Brain(seshat)
st = brain.status()
print(f"✅ seshat جاهز (نموذج نصي) — بحث النت: {'شغال' if st['search'] else 'واقف'} | "
      f"مخ خارجي: {st['api_model'] if st['api'] else 'مش مفعل'} | "
      f"الصور: {st['images_engine']} | "
      f"اتعلم {st['learned']} حاجة قبل كده")


@app.route("/")
def index():
    return render_template("index.html", palette=drawings.PALETTE, grid=drawings.N)


@app.route("/models")
def models_route():
    """List only model versions that have a registered runtime."""
    return jsonify({"models": list_models(), "default": "seshat-1.0"})


@app.route("/info")
def info():
    out = seshat.info()
    out["brain"] = brain.status()
    return jsonify(out)


@app.route("/config", methods=["GET", "POST"])
def config_route():
    """GET يعرض الإعدادات العامة؛ POST يتطلب Authorization: Bearer."""
    if request.method == "GET":
        out = brain.cfg.public()
        out["learned"] = len(brain.learned.items)
        return jsonify(out)

    if not _authorized_config_write():
        # لا نكشف هل التوكن غير مضبوط أم غير صحيح.
        return jsonify({"error": "غير مصرح بتعديل الإعدادات."}), 401

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "محتوى JSON غير صالح."}), 400
    allowed = ("api_key", "api_base", "api_model", "search_enabled")
    brain.cfg.save(**{k: data[k] for k in allowed if k in data})
    out = brain.cfg.public()
    out["learned"] = len(brain.learned.items)
    return jsonify(out)


@app.route("/chat", methods=["POST"])
def chat():
    """الطلب: {"message": "..."} — النموذج نصي بس: مفيش صور ولا فيديو."""
    if not _chat_rate_limit():
        return jsonify({"error": "طلبات كثيرة. انتظر قليلًا ثم حاول مرة أخرى."}), 429

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "محتوى JSON غير صالح."}), 400

    raw_msg = data.get("message", "")
    if not isinstance(raw_msg, str):
        return jsonify({"error": "يجب أن تكون الرسالة نصًا."}), 400
    msg = raw_msg.strip()
    try:
        max_chars = max(100, min(20000, int(os.environ.get("SESHAT_MAX_MESSAGE_CHARS", "8000"))))
    except ValueError:
        max_chars = 8000
    if len(msg) > max_chars:
        return jsonify({"error": f"الرسالة أطول من الحد المسموح ({max_chars} حرف)."}), 413

    t0 = time.time()

    if not msg:
        return jsonify({"response": "اكتب حاجة الأول 🙂", "thinking": "", "confidence": 1.0,
                        "seconds": 0, "fallback": False, "source": "none", "matched": "",
                        "suggestion": ""})
    try:
        temperature = min(1.5, max(0.0, float(data.get("temperature", 0.0))))
    except (TypeError, ValueError):
        temperature = 0.0
    try:
        if os.environ.get("SESHAT_LOCAL_ONLY", "0") != "1":
            out = brain.reply(msg, temperature=temperature)
        else:
            out = seshat.reply(msg, temperature=temperature)
    except Exception:
        app.logger.exception("حصلت مشكلة وقت معالجة طلب الشات")
        return jsonify({"response": "حصلت مشكلة جوه السيرفر. جرب تاني بعد شوية.",
                        "thinking": "", "confidence": 0.0, "seconds": 0, "fallback": True,
                        "source": "none", "matched": "", "suggestion": ""}), 500
    
    # Optional: Enhance with Nawah (البنية التحتية المتقدمة)
    if os.environ.get("SESHAT_USE_NAWAH", "0") == "1" and out.get("response"):
        try:
            from nawah_integration import get_nawah_brain
            nawah_brain = get_nawah_brain()
            enhanced = nawah_brain.enhance_reply(msg, out.get("response", ""), temperature, is_fallback=bool(out.get("fallback")))
            out["response"] = enhanced["reply"]
            out["nawah_analysis"] = enhanced["analysis"]
        except Exception as e:
            app.logger.warning(f"Nawah enhancement failed: {e}")

    return jsonify(out)


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "الملف أو الرسالة أكبر من الحد المسموح."}), 413


# ---------------- routes لنواة-1 (البنية التحتية المتقدمة) ----------------

@app.route("/nawah-status")
def nawah_status():
    try:
        from nawah_integration import get_nawah_brain
        nb = get_nawah_brain()
        st = nb.get_stats()
        st["ready"] = True
        st["enabled_in_chat"] = os.environ.get("SESHAT_USE_NAWAH", "0") == "1"
        return jsonify(st)
    except Exception as e:
        return jsonify(ready=False, error=str(e)), 500


@app.route("/nawah-analyze", methods=["POST"])
def nawah_analyze():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or data.get("question") or "").strip()
    if not text:
        return jsonify(error="اكتب نصًا أو سؤالًا للتحليل أولًا"), 400
    try:
        temperature = min(1.5, max(0.0, float(data.get("temperature", 0.75))))
    except (TypeError, ValueError):
        temperature = 0.75
    try:
        from nawah_integration import get_nawah_brain
        nb = get_nawah_brain()
        result = nb.analyze_only(text, temperature=temperature, save_to_db=True)
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 شغال على http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
