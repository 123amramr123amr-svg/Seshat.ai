# -*- coding: utf-8 -*-
"""
imagegen.py — إنشاء صور واقعية من وصف، وتعديلها بالوصف
======================================================
النموذج المحلي (1.3 مليون معامل) ميقدرش يولّد صور واقعية — بيرسم بالمربعات بس
(data/drawings.py). فالصور الواقعية بتيجي من خدمة صور خارجية، والسيرفر هو اللي بيجيبها
وبيخزنها عنده ويعرضها من عنوانه هو (/generated/<id>.jpg) — المتصفح مبيكلمش الخدمة خالص.

الخطوات لكل صورة:
  1. الوصف بيتترجم إنجليزي (الخدمة بتفهم الإنجليزي أحسن بكتير — ده كان سبب الصور الغلط).
  2. طلب واحد للخدمة ببذرة (seed) ثابتة متخزنة مع الصورة.
  3. الرد بيتراجع: لازم يكون ملف صورة فعلًا (JPEG / PNG / WebP)، مش رسالة خطأ أو كارت
     "الحد المجاني خلص".
  4. الصورة بتتخزن في data/generated/ ومعاها الوصف والبذرة.

التعديل بالوصف ("خليها بالليل"، "ضيف قبعة") = صورة جديدة بنفس البذرة والوصف القديم + التعديل.
ده مش تعديل على نفس البكسلات: الخدمة بتولّد من الأول، فتفاصيل تانية ممكن تتغير.

حدود الخدمة المجانية من غير مفتاح (image.pollinations.ai):
  • صورة واحدة كل 45 ثانية لكل IP — فالطلبات بتستنى دورها هنا بدل ما تترفض.
  • كل صورة عليها لوجو صغير للخدمة في الركن. اللوجو بيتشال بمفتاح مجاني من الخدمة
    (SESHAT_IMAGE_KEY) — مبنقصّوش من الصورة.

مفيش مكتبات صور هنا (ولا PIL): السيرفر بيخزن الملف زي ما هو والمتصفح هو اللي بيفتحه.
"""

import json
import os
import random
import re
import secrets
import threading
import time
from urllib.parse import quote

ROOT = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(ROOT, "data", "generated")

FREE_URL = "https://image.pollinations.ai/prompt/"
KEY_URL = "https://gen.pollinations.ai/image/"
TEXT_URL = "https://text.pollinations.ai/"
MEMORY_URL = "https://api.mymemory.translated.net/get"

SIZE = 768                   # أكبر مقاس في الخدمة المجانية
FREE_GAP = 46.0              # ثواني بين صورتين من غير مفتاح (حد الخدمة 45)
MAX_WAIT = 120.0             # أقصى انتظار للدور قبل ما نقول "مشغول"
KEEP_FILES = 40              # آخر كام صورة تفضل متخزنة
ID_RE = re.compile(r"^[a-f0-9]{16}$")

_STYLE_WORDS = re.compile(
    r"\b(photo|photograph|realistic|photorealistic|cartoon|anime|manga|painting|oil|watercolor|"
    r"drawing|sketch|pencil|3d|render|pixel|logo|icon|illustration|vector|comic|clay|"
    r"black and white|line art|poster)\b", re.I)
_ARABIC = re.compile(r"[\u0600-\u06FF]")


def is_arabic(text):
    return bool(_ARABIC.search(text or ""))


def _default_get(url, headers=None, timeout=60):
    """بيرجع (الكود، الهيدرز، المحتوى). أي مشكلة شبكة → (0, {}, b"")"""
    import requests
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        return r.status_code, dict(r.headers), r.content
    except Exception:
        return 0, {}, b""


def image_kind(data):
    """نوع ملف الصورة من أول بايتات، أو None لو مش صورة"""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def add_style(prompt_en):
    """لو الوصف مفيهوش أسلوب (كرتون، زيتي…) بنطلب صورة واقعية — ده اللي المستخدم طالبه"""
    prompt_en = re.sub(r"\s+", " ", prompt_en).strip(" .,،")
    if _STYLE_WORDS.search(prompt_en):
        return prompt_en
    return prompt_en + ", realistic photo, sharp focus, detailed"


class ImageMaker:
    """
    بيدير طلبات الصور: كل طلب ليه id وملف وصف (json)، والصورة نفسها بتتولد أول مرة حد
    يطلب ملفها (ensure) — فرد الشات بيرجع فورًا والصورة بتيجي لما تجهز.
    """

    def __init__(self, store_dir=STORE_DIR, key="", http_get=None, llm=None, sleep=time.sleep,
                 clock=time.time):
        self.dir = store_dir
        self.key = key or ""
        self.get = http_get or _default_get
        self.llm = llm                    # دالة(نص) → ترجمة من المخ الخارجي لو متربط، أو None
        self.sleep = sleep
        self.clock = clock
        self._lock = threading.Lock()     # صورة واحدة في المرة
        self._next_free = 0.0             # أقرب وقت مسموح فيه بصورة مجانية
        self._waiting = 0

    # ---------- التخزين ----------
    def _meta_path(self, image_id):
        return os.path.join(self.dir, image_id + ".json")

    def file_path(self, image_id):
        return os.path.join(self.dir, image_id + ".jpg")

    def meta(self, image_id):
        if not isinstance(image_id, str) or not ID_RE.match(image_id):
            return None
        try:
            with open(self._meta_path(image_id), encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save(self, meta):
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._meta_path(meta["id"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, self._meta_path(meta["id"]))

    def _cleanup(self):
        try:
            metas = sorted((f for f in os.listdir(self.dir) if f.endswith(".json")),
                           key=lambda f: os.path.getmtime(os.path.join(self.dir, f)))
        except OSError:
            return
        for name in metas[:-KEEP_FILES]:
            for path in (os.path.join(self.dir, name),
                         os.path.join(self.dir, name[:-5] + ".jpg")):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ---------- طلب جديد ----------
    def new(self, prompt, parent=None, edit=None, seed=None):
        """
        بيسجل طلب صورة وبيرجع الـ id (الصورة لسه متولدتش).
        parent + edit: تعديل بالوصف على صورة قديمة (نفس البذرة، والوصف = القديم + التعديل).
        parent من غير edit: صورة تانية لنفس الوصف (بذرة جديدة).
        """
        base = self.meta(parent) if parent else None
        if base and edit:
            prompt = base["prompt"] + "، " + edit if is_arabic(base["prompt"] + edit) \
                else base["prompt"] + ", " + edit
            seed = base["seed"]
        elif base:
            prompt = base["prompt"]
        meta = {"id": secrets.token_hex(8), "prompt": prompt.strip(), "prompt_en": "",
                "seed": int(seed if seed is not None else random.randint(1, 2_000_000_000)),
                "state": "pending", "error": "", "created": int(self.clock()),
                "parent": base["id"] if base else "", "edit": edit or "",
                "arabic": is_arabic(prompt + (edit or ""))}
        if base and edit and base.get("prompt_en"):
            meta["base_en"] = base["prompt_en"]
        self._save(meta)
        self._cleanup()
        return meta["id"]

    # ---------- الترجمة ----------
    def translate(self, text):
        """
        وصف عربي → إنجليزي. بالترتيب: المخ الخارجي لو متربط، نموذج نصوص الخدمة، خدمة ترجمة.
        بيرجع None لو كله فشل (ساعتها الوصف بيتبعت زي ما هو).
        """
        text = text.strip()
        if not is_arabic(text):
            return text
        ask = ("Translate this Egyptian Arabic image description into one short English image "
               "prompt. Reply with the English prompt only, no quotes, no explanation: " + text)
        if self.llm:
            try:
                out = self._clean(self.llm(ask))
                if out:
                    return out
            except Exception:
                pass
        code, _, body = self.get(TEXT_URL + quote(ask, safe=""), timeout=40)
        if code == 200:
            out = self._clean(body.decode("utf-8", "replace"))
            if out:
                return out
        code, _, body = self.get(MEMORY_URL + "?q=" + quote(text[:450], safe="") + "&langpair=ar|en",
                                 timeout=20)
        if code == 200:
            try:
                out = self._clean(json.loads(body.decode("utf-8"))["responseData"]["translatedText"])
                if out:
                    return out
            except Exception:
                pass
        return None

    def rewrite(self, base, base_en, edit):
        """
        تعديل بالوصف: بيكتب وصف إنجليزي جديد للصورة بعد التعديل ("عربية حمرا" + "خليها زرقا"
        → "a blue car"). محتاج نموذج نصوص؛ لو مش متاح بيرجع None والتعديل بيتضاف على الوصف.
        """
        ask = ("Original image description: " + (base_en or base) + "\n"
               "Requested change (may be Egyptian Arabic): " + edit + "\n"
               "Write one short English image prompt that describes the image after applying "
               "the change. Keep everything else the same. Reply with the prompt only, no "
               "quotes, no explanation.")
        if self.llm:
            try:
                out = self._clean(self.llm(ask))
                if out:
                    return out
            except Exception:
                pass
        code, _, body = self.get(TEXT_URL + quote(ask, safe=""), timeout=40)
        if code == 200:
            return self._clean(body.decode("utf-8", "replace"))
        return None

    @staticmethod
    def _clean(text):
        """ترجمة سليمة = سطر إنجليزي قصير من غير عربي"""
        text = (text or "").strip().strip("\"'“”`").strip()
        text = text.split("\n")[0].strip()
        if not text or len(text) > 400 or is_arabic(text) or not re.search(r"[A-Za-z]{2}", text):
            return None
        return text

    # ---------- التوليد ----------
    def _url(self, prompt_en, seed):
        base = KEY_URL if self.key else FREE_URL
        return "%s%s?width=%d&height=%d&seed=%d" % (base, quote(prompt_en[:400], safe=""),
                                                    SIZE, SIZE, seed)

    def _fetch(self, prompt_en, seed):
        """بيرجع (بايتات الصورة، None) أو (None، سبب: busy / refused / down)"""
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        for attempt in range(2):
            if not self.key:
                wait = self._next_free - self.clock()
                if wait > 0:
                    self.sleep(wait)
            code, head, body = self.get(self._url(prompt_en, seed), headers=headers, timeout=90)
            if not self.key:
                self._next_free = self.clock() + FREE_GAP
            if code == 200 and image_kind(body):
                return body, None
            if code == 402 and attempt == 0 and not self.key:
                # الحد المجاني: الخدمة بتقول نستنى قد إيه
                try:
                    retry = float({k.lower(): v for k, v in head.items()}.get("retry-after", 45))
                except ValueError:
                    retry = 45.0
                self._next_free = self.clock() + min(60.0, max(1.0, retry)) + 1.0
                continue
            if code in (402, 429):
                return None, "busy"
            if code in (400, 403, 422, 451):
                return None, "refused"
            return None, "down"
        return None, "busy"

    def ensure(self, image_id):
        """
        بيتأكد إن الصورة متولدة. بيرجع (مسار الملف، None) أو (None، رسالة للمستخدم).
        آمن من كذا طلب في نفس الوقت: واحد بس بيولّد والباقي بيستنوا.
        """
        meta = self.meta(image_id)
        if meta is None:
            return None, "الصورة دي مش موجودة."
        if meta["state"] == "ready" and os.path.exists(self.file_path(image_id)):
            return self.file_path(image_id), None
        if meta["state"] == "failed":
            return None, meta["error"]

        # لو الطابور طويل نقول مشغول بدل ما الطلب يفضل متعلق
        queue = max(0.0, self._next_free - self.clock()) + self._waiting * FREE_GAP
        if not self.key and queue > MAX_WAIT:
            return None, self._message("busy", meta["arabic"])
        self._waiting += 1
        try:
            with self._lock:
                meta = self.meta(image_id) or meta          # ممكن طلب تاني خلّصها واحنا مستنيين
                if meta["state"] == "ready" and os.path.exists(self.file_path(image_id)):
                    return self.file_path(image_id), None
                if meta["state"] == "failed":
                    return None, meta["error"]
                if not meta["prompt_en"]:
                    if meta.get("edit") and meta.get("parent"):
                        # تعديل: وصف جديد بعد تطبيق التعديل. لو مفيش نموذج نصوص متاح، الوصف
                        # كله (القديم + التعديل) بيتترجم مع بعض عشان المعنى ميضيعش.
                        base_en = re.sub(r", realistic photo, sharp focus, detailed$", "",
                                         meta.get("base_en", ""))
                        base_ar = meta["prompt"][:-len(meta["edit"])].rstrip("،, ")
                        new_en = self.rewrite(base_ar, base_en, meta["edit"])
                        meta["rewritten"] = bool(new_en)
                        meta["prompt_en"] = add_style(new_en or self.translate(meta["prompt"])
                                                      or meta["prompt"])
                    else:
                        meta["prompt_en"] = add_style(self.translate(meta["prompt"])
                                                      or meta["prompt"])
                    self._save(meta)
                started = self.clock()
                data, why = self._fetch(meta["prompt_en"], meta["seed"])
                meta["seconds"] = round(self.clock() - started, 1)     # انتظار الدور + التوليد
                if data is None:
                    if why != "refused":
                        # مشغول أو الشبكة واقعة = مؤقت: الطلب يفضل pending عشان "جرب تاني" يشتغل
                        return None, self._message(why, meta["arabic"])
                    meta["state"], meta["error"] = "failed", self._message(why, meta["arabic"])
                    self._save(meta)
                    return None, meta["error"]
                with open(self.file_path(image_id), "wb") as f:
                    f.write(data)
                meta["state"], meta["kind"] = "ready", image_kind(data)
                self._save(meta)
                return self.file_path(image_id), None
        finally:
            self._waiting -= 1

    @staticmethod
    def _message(why, arabic):
        if why == "busy":
            return ("مولّد الصور مشغول دلوقتي (بيطلع صورة واحدة كل 45 ثانية). استنى دقيقة ودوس "
                    "«جرب تاني»." if arabic else
                    "The image generator is busy right now (one image every 45 seconds). "
                    "Wait a minute and press “Try again”.")
        if why == "refused":
            return ("مولّد الصور رفض الوصف ده. جرب وصف تاني." if arabic else
                    "The image generator refused this description. Try another one.")
        return ("معرفتش أوصل لمولّد الصور دلوقتي. جرب تاني بعد شوية، أو اكتب «ارسم بالمربعات …» "
                "وأنا أرسمها بنفسي." if arabic else
                "I could not reach the image generator right now. Try again later, or write "
                "“draw in pixels …” and I will draw it myself.")
