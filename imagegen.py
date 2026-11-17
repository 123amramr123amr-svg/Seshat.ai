# -*- coding: utf-8 -*-
"""
imagegen.py — إنشاء الصور من خوفو (النموذج الموحّد seshat 1.0)
==============================================================
جولة خ8 (التوحيد): كل صور بتتطلبه من seshat بتتولد محليًا 100% من **خوفو** —
نموذج الانتشار (DDPM) + الرسم الإجرائي الهجين بتاع المستودع نفسه. مفيش أي
خدمة صور خارجية، مفيش نت، مفيش لوجو، مفيش انتظار 45 ثانية، ومفيش مفاتيح.

العقد القديم ما اتكسرش (عشان المخ والواجهة والاختبارات ماشية زي ما هي):
  • ImageMaker.new(prompt, parent, edit, seed) → id، والرد بيرجع فورًا.
  • ImageMaker.ensure(id) → (مسار الملف، None) أو (None، رسالة للمستخدم) —
    أول مرة بيتولّد فعليًا: خوفو بيفهم الوصف العربي/الإنجليزي مباشرة
    (مرادفات + تركيبات مركبة + دلوّات دلالية) وبرسم على 750×750.
  • ImageMaker.meta(id) → بيانات الصورة.
  • image_kind(data) → نوع الملف من أول بايتات.
  • add_style(prompt) → إثراء الوصف (مازال موجود للتوافق).

الصور بتتخزن data/generated/<id>.png ومعاها meta json فيها الوصف والبذرة.
التعديل بالوصف («خليها بالليل») = وصف جديد بنفس البذرة — خوفو حتمي: نفس
البذرة ونفس الوصف = نفس الصورة بالحرف، ووصف مختلف = مشهد مختلف مقصود.
"""

import json
import os
import random
import re
import secrets
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STORE_DIR = os.path.join(ROOT, "data", "generated")

SIZE = 750                   # أعلى دقة في خوفو (الرسم الهجين خ5)
STEPS = 50                   # خطوات جودة وسط — توازن سرعة/وضوح (خ8)
KEEP_FILES = 40              # آخر كام صورة تفضل متخزنة
ID_RE = re.compile(r"^[a-f0-9]{16}$")

_ARABIC = re.compile(r"[\u0600-\u06FF]")


def is_arabic(text):
    return bool(_ARABIC.search(text or ""))


def image_kind(data):
    """نوع ملف الصورة من أول بايتات، أو None لو مش صورة"""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def add_style(prompt):
    """إثراء الوصف بمؤشرات وضوح — بيضل جزء من meta عشان التوافق والتوثيق."""
    prompt = re.sub(r"\s+", " ", prompt or "").strip(" .,،")
    return prompt


def _khufu():
    """تحميل مولّد خوفو من مجلده — نفس طريقة app.py (مفيش تعارض أسماء)."""
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    kdir = os.path.join(here, "khufu")
    if kdir not in sys.path:
        sys.path.insert(0, kdir)
    import generate as khufu_generate
    return khufu_generate


class ImageMaker:
    """
    بيدير طلبات الصور من خوفو: كل طلب ليه id وملف وصف (json)، والصورة بتتولد أول
    مرة حد يطلب ملفها (ensure) — فرد الشات بيرجع فورًا والرسم بيتعمل خلف الكواليس.
    خوفو بيتسلسل here: صورة واحدة في المرة عشان الـ CPU، والباقي بيستنى على القفل.
    """

    def __init__(self, store_dir=STORE_DIR, size=SIZE, steps=STEPS, **_legacy):
        # _legacy: مفاتيح قديمة (key/http_get/sleep/clock/llm) بتتقبل وتتتجاهل —
        # التوليد بقى محلي بالكامل من غير شبكة ولا مفاتيح.
        self.dir = store_dir
        self.size = int(size)
        self.steps = int(steps)
        self._lock = threading.Lock()

    # ---------- التخزين ----------
    def _meta_path(self, image_id):
        return os.path.join(self.dir, image_id + ".json")

    def file_path(self, image_id):
        return os.path.join(self.dir, image_id + ".png")

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
                         os.path.join(self.dir, name[:-5] + ".png"),
                         os.path.join(self.dir, name[:-5] + ".jpg")):
                try:
                    os.remove(path)
                except OSError:
                    pass

    # ---------- طلب جديد ----------
    def new(self, prompt, parent=None, edit=None, seed=None, size=None, steps=None):
        """
        بيسجل طلب صورة وبيرجع الـ id (الصورة لسه متولدتش).
        parent + edit: وصف جديد بنفس البذرة (تعديل بالوصف على صورة قديمة).
        parent من غير edit: صورة تانية لنفس الوصف (بذرة جديدة).
        """
        base = self.meta(parent) if parent else None
        if base and edit:
            prompt = base["prompt"] + "، " + edit if is_arabic(base["prompt"] + edit) \
                else base["prompt"] + ", " + edit
            seed = base["seed"]
        elif base:
            prompt = base["prompt"]
        meta = {"id": secrets.token_hex(8), "prompt": add_style(prompt),
                "seed": int(seed if seed is not None else random.randint(1, 2_000_000_000)),
                "state": "pending", "error": "", "created": int(time.time()),
                "parent": base["id"] if base else "", "edit": edit or "",
                "arabic": is_arabic(prompt + (edit or "")),
                "engine": "khufu", "model": "seshat 1.0",
                "size": int(size or self.size), "steps": int(steps or self.steps)}
        if base and edit:
            meta["base_prompt"] = base.get("base_prompt") or base["prompt"]
        self._save(meta)
        self._cleanup()
        return meta["id"]

    # ---------- التوليد ----------
    def _message(self, why, arabic):
        if why == "khufu-missing":
            return ("مولّد خوفو مش محمّل دلوقتي — جرب تاني بعد إعادة تشغيل السيرفر." if arabic
                    else "Khufu is not loaded right now — restart the server and try again.")
        return ("حصلت مشكلة وأنا بديرسم — جرب تاني." if arabic
                else "Something went wrong while drawing — try again.")

    def ensure(self, image_id):
        """
        بيتأكد إن الصورة متولدة من خوفو. بيرجع (مسار الملف، None) أو
        (None، رسالة للمستخدم). آمن من كذا طلب في نفس الوقت: واحد بس بيرسم
        والباقي بيستنى — ومن غير أي اتصال شبكة خالص.
        """
        meta = self.meta(image_id)
        if meta is None:
            return None, "الصورة دي مش موجودة."
        if meta["state"] == "ready" and os.path.exists(self.file_path(image_id)):
            return self.file_path(image_id), None
        if meta["state"] == "failed":
            return None, meta["error"]

        with self._lock:
            meta = self.meta(image_id) or meta          # ممكن طلب تاني خلّصها واحنا مستنيين
            if meta["state"] == "ready" and os.path.exists(self.file_path(image_id)):
                return self.file_path(image_id), None
            if meta["state"] == "failed":
                return None, meta["error"]
            try:
                kh = _khufu()
                img, _cid = kh.generate(meta["prompt"], size=meta["size"],
                                        steps=meta["steps"], seed=meta["seed"])
                path = self.file_path(image_id)
                img.save(path)
                with open(path, "rb") as f:
                    kind = image_kind(f.read(16))
                meta["state"], meta["kind"] = "ready", kind or "png"
                meta["resolved"] = _cid                  # الصنف/التركيبة اللي خوفو فهمها
                self._save(meta)
                return path, None
            except Exception as e:                        # خوفو واقع = فشل صريح مع رسالة واضحة
                meta["state"], meta["error"] = "failed", self._message("error", meta["arabic"])
                self._save(meta)
                return None, meta["error"] + (" (%s)" % type(e).__name__)
