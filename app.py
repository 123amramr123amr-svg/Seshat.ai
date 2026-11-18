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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # خ12: مستندات لحد ٢م حرف (UTF-8)

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
    
    # ========================================
    # الأدوات المتقدمة (Advanced Tools)
    # ========================================
    try:
        from tools_integration import detect_tool_request, execute_tool_request
        
        tool_request = detect_tool_request(msg)
        if tool_request:
            tool_response = execute_tool_request(tool_request)
            return jsonify({
                "response": tool_response,
                "thinking": f"استخدمت أداة: {tool_request['tool']}",
                "confidence": 1.0,
                "seconds": time.time() - t0,
                "fallback": False,
                "source": "tool",
                "matched": tool_request['tool'],
                "suggestion": ""
            })
    except Exception as e:
        app.logger.warning(f"Tools detection failed: {e}")
    
    # ========================================
    # الرد العادي (Normal Reply)
    # ========================================
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


@app.route("/api/long", methods=["POST"])
def api_long():
    """
    خ12 — السياق الممتد: مستند ضخم (لحد ٢٠٠٠٠٠٠ حرف) + سؤال عنه.
    الطلب: {"document": "...", "question": "..."}
    المستند بيتدفق كله في ذاكرة النافذة المنزلقة، والرد بيتولد من السياق الأخير.
    """
    if not _chat_rate_limit():
        return jsonify({"error": "طلبات كثيرة. انتظر قليلًا ثم حاول مرة أخرى."}), 429
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "محتوى JSON غير صالح."}), 400
    doc = data.get("document", "")
    q = data.get("question", "")
    if not isinstance(doc, str) or not isinstance(q, str):
        return jsonify({"error": "document و question لازم يكونوا نص."}), 400
    cap = int(seshat.info().get("max_context", 2_000_000))
    if len(doc) > cap:
        return jsonify({"error": f"المستند أطول من السياق المسموح ({cap} حرف)."}), 413
    if not q.strip():
        return jsonify({"error": "ابعت سؤال مع المستند."}), 400
    try:
        temperature = min(1.5, max(0.0, float(data.get("temperature", 0.0))))
    except (TypeError, ValueError):
        temperature = 0.0
    try:
        out = seshat.reply_long(doc, q, temperature=temperature)
    except Exception:
        app.logger.exception("مشكلة في معالجة المستند الطويل")
        return jsonify({"error": "حصلت مشكلة جوه السيرفر. جرب تاني بعد شوية."}), 500
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


# ========================================
# أدوات seshat المتقدمة (Advanced Tools)
# ========================================

@app.route("/tools/browse", methods=["POST"])
def tools_browse():
    """تصفح الويب المتقدم"""
    data = request.get_json(silent=True) or {}
    url = str(data.get("url") or "").strip()
    if not url:
        return jsonify(error="الرابط مطلوب"), 400
    
    if not url.startswith('http'):
        url = 'https://' + url
    
    try:
        from tools import browse_web
        result = browse_web(
            url,
            extract_tables=bool(data.get("extract_tables", True)),
            follow_links=bool(data.get("follow_links", False)),
            max_links=int(data.get("max_links", 3))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/tools/execute", methods=["POST"])
def tools_execute():
    """تشغيل الأكواد"""
    data = request.get_json(silent=True) or {}
    code = str(data.get("code") or "").strip()
    if not code:
        return jsonify(error="الكود مطلوب"), 400
    
    try:
        from tools import execute_code
        result = execute_code(
            code,
            language=str(data.get("language", "python")),
            timeout=int(data.get("timeout", 10))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/tools/file", methods=["POST"])
def tools_create_file():
    """إنشاء ملفات (Text/PDF)"""
    data = request.get_json(silent=True) or {}
    content = str(data.get("content") or "").strip()
    if not content:
        return jsonify(error="المحتوى مطلوب"), 400
    
    file_type = str(data.get("type", "text")).lower()
    
    try:
        if file_type == "pdf":
            from tools import create_pdf_file
            result = create_pdf_file(
                content,
                title=str(data.get("title", "مستند")),
                filename=data.get("filename")
            )
        else:
            from tools import create_text_file
            result = create_text_file(
                content,
                filename=data.get("filename")
            )
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/tools/memory", methods=["GET", "POST", "DELETE"])
def tools_memory():
    """الذاكرة طويلة المدى"""
    try:
        from tools import tools
        
        if request.method == "GET":
            # جلب الذكريات
            limit = int(request.args.get("limit", 100))
            memories = tools.memory.get_all(limit=limit)
            return jsonify(memories=memories, count=len(memories))
        
        elif request.method == "POST":
            # حفظ ذكرى جديدة
            data = request.get_json(silent=True) or {}
            content = str(data.get("content") or "").strip()
            if not content:
                return jsonify(error="المحتوى مطلوب"), 400
            
            memory = tools.remember(
                content,
                tags=data.get("tags"),
                importance=int(data.get("importance", 5))
            )
            return jsonify(memory)
        
        elif request.method == "DELETE":
            # حذف ذكرى
            data = request.get_json(silent=True) or {}
            memory_id = str(data.get("id") or "").strip()
            if not memory_id:
                return jsonify(error="المعرف مطلوب"), 400
            
            success = tools.memory.forget(memory_id)
            return jsonify(success=success)
    
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/tools/recall", methods=["POST"])
def tools_recall():
    """استرجاع ذكريات"""
    data = request.get_json(silent=True) or {}
    query = str(data.get("query") or "").strip()
    if not query:
        return jsonify(error="الاستعلام مطلوب"), 400
    
    try:
        from tools import tools
        memories = tools.recall(
            query,
            limit=int(data.get("limit", 5))
        )
        return jsonify(memories=memories, count=len(memories))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/tools/image", methods=["POST"])
def tools_understand_image():
    """فهم الصور"""
    # دعم رفع الملفات أو مسار الملف
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify(error="مفيش ملف"), 400
        
        # حفظ الملف مؤقتاً
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.gettempdir()) / 'seshat_uploads'
        temp_dir.mkdir(exist_ok=True)
        
        filename = f"{int(time.time())}_{file.filename}"
        filepath = temp_dir / filename
        file.save(str(filepath))
        
        image_path = str(filepath)
    else:
        data = request.get_json(silent=True) or {}
        image_path = str(data.get("path") or "").strip()
        if not image_path:
            return jsonify(error="المسار أو الملف مطلوب"), 400
    
    try:
        from tools import understand_image
        result = understand_image(image_path)
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.route("/download/<path:filename>")
def download_file(filename):
    """تحميل ملف من data/files"""
    from pathlib import Path
    
    filepath = Path('data/files') / filename
    if not filepath.exists():
        return jsonify(error="الملف غير موجود"), 404
    
    return send_file(str(filepath), as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 شغال على http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
