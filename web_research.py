# -*- coding: utf-8 -*-
"""خ33 — وضع البحث في الويب وتحليل مواقع متعددة (بأمر المستخدم: «يقدر يدخل
على أكثر من موقع أو أكثر من مكان ويحلل المعلومات ويجيب رد مفصل»).

الاتجاه واحد: ويب → نص منظم. ممنوع توليد الصور نهائيًا (حكم خ10).
صفر مكتبات جديدة: requests + مكتبة بايثون القياسية بس (قاعدة المشروع).

المعمارية:
1) بحث: Google News RSS + Bing News RSS (اتجربوا فعليًا: 100 و8 عناصر) —
   محركات البحث HTML محجوبة من بيئة السحابة (DuckDuckGo/Bing صفحة تحدي،
   Wikipedia 403) فالنتيجة الصادقة: RSS هو المسار الفعلي للبحث.
2) قراءة مباشرة متعددة المواقع: كل رابط من نتائج البحث أو من رسالة
   المستخدم بيتقري فعليًا (متجرب: BBC عربي 200/590KB) والنص بيتستخلص
   بـHTMLParser القياسي بدون مكتبات.
3) تحليل: جمل كل موقع بتتقيم بتداخل كلمات مفتاحية السؤال، والرد المفصل
   بيتجمع: ملخص + تحليل لكل موقع + استنتاج + مصادر مرقمة + إحصائيات.
"""
import re
import time
import requests
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse, unquote, quote_plus

_UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
       "Accept-Language": "ar,en;q=0.9"}
_TIMEOUT = 12
_MAX_CHARS = 15000          # سقف نص الصورة الواحدة من موقع
_MAX_SITES = 4              # عدد المواقع المقراءة فعليًا في التحليل

# ---------- كلمات النية ----------
_INTENT_RE = re.compile(
    r"(ابحث\s+في\s+الويب|ابحث\s+عن|ابحث\s+ايع|ابحث|بحث\s+عن|بحث\s+في\s+الويب|"
    r"حلل\s+(?:ال)?(?:روابط|الموقع|المواقع|لينك)|اقرأ\s+(?:ال)?(?:روابط|الموقع|المواقع|لينك)|"
    r"جيب\s+رد\s+مفصل|كولي\s+تفاصيل|search\s+for|search\s+web|web\s+search)", re.I)
_MEMORY_HINT = re.compile(r"(ذاكرتك|الذاكرة|في\s+الذاكرة|memory)", re.I)
_URL_RE = re.compile(r"https?://[^\s\)\]»\"']+", re.I)


def detect_intent(message):
    """كشف نية بحث ويب صريحة — الرسايل العادية ما بتتلمسش."""
    m = (message or "").strip()
    if not m or len(m) > 300:
        return False
    if _MEMORY_HINT.search(m):
        return False            # «ابحث في ذاكرتك» مش ويب
    if _URL_RE.search(m):
        return True             # رابط مباشر = نية تحليل ويب
    return bool(_INTENT_RE.search(m))


def extract_query(message):
    """استخراج الاستعلام من الرسالة + الروابط المباشرة لو فيه — مع تنظيف
    كلمات اللفظ والضمائر المرفقة («حلل الروابط دي عن أخبار مصر» → «أخبار مصر»)."""
    m = (message or "").strip()
    urls = _URL_RE.findall(m)
    q = _URL_RE.sub(" ", m)
    q = _INTENT_RE.sub(" ", q)
    q = re.sub(r"(?:ال)?روابط\s*(?:دي|ده|هذه|هذول|بتاع)?\s*", " ", q)
    q = re.sub(r"(?:ال)?موقع\s*(?:ده|دي|هذا)?\s*", " ", q)
    q = re.sub(r"\b(?:دي|ده|دا|هذه|هذا|هذول)\b", " ", q)
    q = re.sub(r"^\s*(?:عن|ان|أن)\s+", " ", q)
    q = re.sub(r"^[\s:،,-]+|[\s:،,-]+$", "", q)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q, urls[:6]          # أقصى 6 روابط مباشرة


# ---------- استخلاص نص HTML ----------
class _TextExtract(HTMLParser):
    """استخلاص نص نظيف من HTML: تجاهل السكريبتات والأنماط والترويسة والقوائم."""

    _SKIP = {"script", "style", "noscript", "header", "footer", "nav",
             "aside", "svg", "form", "iframe", "select", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in ("p", "br", "li", "div", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)


def html_to_text(html):
    p = _TextExtract()
    try:
        p.feed(html)
        p.close()
    except Exception:
        pass
    t = unescape(" ".join(p.parts))
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def fetch_page(url):
    """قراءة موقع واحد فعليًا — بيرجع dict (سليم أو فاشل بالسبب) بلا استثناءات."""
    t0 = time.time()
    try:
        r = requests.get(url, headers=_UA, timeout=_TIMEOUT, stream=True)
        ctype = (r.headers.get("content-type") or "").lower()
        if r.status_code != 200:
            return {"ok": False, "url": url, "why": "HTTP %d" % r.status_code, "seconds": round(time.time() - t0, 2)}
        if "html" not in ctype and "xml" not in ctype and "json" not in ctype and "rss" not in ctype and "text" not in ctype:
            return {"ok": False, "url": url, "why": "نوع محتوى غير نصي: %s" % ctype.split(";")[0],
                    "seconds": round(time.time() - t0, 2)}
        chunk = b""
        for piece in r.iter_content(chunk_size=65536):
            chunk += piece
            if len(chunk) > _MAX_CHARS * 2:
                break
        r.close()
        enc = None
        m = re.search(r"charset=([\w\-]+)", ctype)
        if m:
            enc = m.group(1)
        try:
            text = chunk.decode(enc or "utf-8", errors="replace")
        except LookupError:
            text = chunk.decode("utf-8", errors="replace")
        if "<" in text[:2000]:                 # HTML → نص نظيف
            text = html_to_text(text)
        text = re.sub(r"\s{2,}", " ", text).strip()[:_MAX_CHARS]
        return {"ok": True, "url": url, "text": text, "chars": len(text),
                "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        return {"ok": False, "url": url, "why": str(e)[:120], "seconds": round(time.time() - t0, 2)}


def site_name(url):
    try:
        d = urlparse(url).netloc.replace("www.", "")
        return d or "موقع"
    except Exception:
        return "موقع"


# ---------- البحث ----------
def _parse_google_rss(xml_text):
    """Google News RSS: <item><title/><link/><pubDate/><source/><description/>"""
    out = []
    for item in re.findall(r"<item>(.*?)</item>", xml_text, re.S)[:12]:
        title = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, re.S)
        link = re.search(r"<link>(http[^<]+)</link>", item)
        desc = re.search(r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>", item, re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", item)
        src = re.search(r"<source[^>]*>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</source>", item, re.S)
        if title and link:
            out.append({"title": unescape(title.group(1)).strip(),
                        "url": link.group(1).strip(),
                        "snippet": html_to_text(desc.group(1))[:400] if desc else "",
                        "date": date.group(1).strip() if date else "",
                        "source": src.group(1).strip() if src else site_name(link.group(1))})
    return out


def _parse_bing_rss(xml_text):
    """Bing News RSS: نفس البنية تقريبًا."""
    out = []
    for item in re.findall(r"<item>(.*?)</item>", xml_text, re.S)[:12]:
        title = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", item, re.S)
        link = re.search(r"<link>(http[^<]+)</link>", item)
        desc = re.search(r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>", item, re.S)
        date = re.search(r"<pubDate>(.*?)</pubDate>", item)
        if title and link:
            out.append({"title": unescape(title.group(1)).strip(),
                        "url": link.group(1).strip(),
                        "snippet": html_to_text(desc.group(1))[:400] if desc else "",
                        "date": date.group(1).strip() if date else "",
                        "source": site_name(link.group(1))})
    return out


def _ddg_search(query, max_results=6):
    """بحث DuckDuckGo lite — بنفس توقيع الـPOST الناجح المتوثق في الدماغ
    (_ddg من خ15: POST + Referer — هو الوحيد اللي بيعدّي من حجب السحابة).
    النتايج روابط أصلية مباشرة قابلة للقراءة الفعلية → direct=True."""
    out = []
    try:
        r = requests.post("https://lite.duckduckgo.com/lite/", data={"q": query},
                          headers={**_UA, "Referer": "https://lite.duckduckgo.com/"},
                          timeout=8)
        links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>\s*(.*?)\s*</a>', r.text)
        snippets = re.findall(r"class=['\"]result-snippet['\"][^>]*>(.*?)</td>", r.text, re.S)
        for href, title in links:
            if not href.startswith("http"):
                m = re.search(r"uddg=([^&]+)", href)
                if not m:
                    continue
                href = unquote(m.group(1))
            if "duckduckgo.com" in urlparse(href).netloc:
                continue
            title = re.sub(r"<[^>]+>", "", title).strip()
            if not title:
                continue
            snip = ""
            if len(out) < len(snippets):
                snip = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", snippets[len(out)])).strip()[:400]
            out.append({"title": title, "url": href, "snippet": snip,
                        "date": "", "source": site_name(href), "direct": True})
            if len(out) >= max_results:
                break
    except Exception:
        pass
    return out


def search_web(query, max_results=8):
    """بحث فعلي بثلاثة مسارات موثقة:
    1) DuckDuckGo lite (POST — بيعدّي من الحجب، روابط أصلية قابلة للقراءة)
    2) Bing News RSS (روابط أصلية بعد فك apiclick)
    3) Google News RSS (مقتطفات فقط — روابطه تحويل JS متجربة ومتاحة للملخص)"""
    results, notes = [], []
    seen = set()

    def _add(items, direct):
        n = 0
        for it in items:
            it["direct"] = direct
            key = it["url"].split("&")[0][:80]
            if key in seen:
                continue
            seen.add(key)
            results.append(it)
            n += 1
        return n

    ddg = _ddg_search(query, max_results=6)
    _add(ddg, True)
    notes.append("DuckDuckGo: %d نتيجة (روابط أصلية قابلة للقراءة)" % len(ddg))
    try:
        r = requests.get("https://www.bing.com/news/search?q=%s&format=RSS" % quote_plus(query),
                         headers=_UA, timeout=_TIMEOUT)
        if r.status_code == 200:
            items = _parse_bing_rss(r.text)
            for it in items:
                it["url"] = _bing_decode(it["url"])
            n = _add(items, True)
            notes.append("Bing News: %d نتيجة (روابط أصلية قابلة للقراءة)" % n)
        else:
            notes.append("Bing News: HTTP %d" % r.status_code)
    except Exception as e:
        notes.append("Bing News: %s" % str(e)[:60])
    try:
        r2 = requests.get("https://news.google.com/rss/search?q=%s&hl=ar&gl=EG&ceid=EG:ar" % quote_plus(query),
                          headers=_UA, timeout=_TIMEOUT)
        if r2.status_code == 200:
            items2 = _parse_google_rss(r2.text)
            n2 = _add(items2, False)
            notes.append("Google News: %d نتيجة (مقتطفات فقط — روابطها تحويل JS)" % n2)
        else:
            notes.append("Google News: HTTP %d" % r2.status_code)
    except Exception as e:
        notes.append("Google News: %s" % str(e)[:60])
    return results[:max_results], notes


# ---------- التحليل والتركيب ----------
_AR_STOP = set("من في على عن الى إلى مع او أو ثم هذا هذه ذلك التي الذي الذي ما لا هوا هو هي كان كانت يكون انت أنت انا أنا الش شفت يقول قول ليه ازاي إزاي دايم كده للا اه صراحة جدا جدًا رجاء لو كل بعض غير بين عند عندنا عندي هن يعني يعني مو مش بس حتى خلاص طب تابع كمان قبل بعد فوق تحت صح غلط قسم".split())
# خ33: كلمات النية نفسها مش مفتاحية — «ابحث عن قناة السويس» المفتاح «قناة السويس»
_INTENT_WORDS = set("ابحث بحث بحثت جيب كولي اكتب حلل اقرا اقرأ دور دورت مواقع موقع روابط رابط ويب انت وحلل وقريت دخلت فعليا فعليًا مطلوب المطلوب".split())
_EN_STOP = set("the a an of in on at to for with and or but is are was were this that these those you i me my your from by as it its be been do does did have has had not no yes can will would should could about more most some any what how why when where who which search find read web sites site links link".split())


def _strip_ar_def(w):
    """فك ال التعريف العربي عشان «السويس» تطابق «سويس» في نص المواقع."""
    return w[2:] if w.startswith("ال") and len(w) > 4 else w


def keywords_of(query):
    words = re.findall(r"[\w\u0600-\u06FF]{3,}", (query or "").lower())
    out = []
    for w in words:
        if w in _AR_STOP or w in _INTENT_WORDS or w in _EN_STOP:
            continue
        w = _strip_ar_def(w)
        if w and w not in _AR_STOP and w not in _INTENT_WORDS and len(w) >= 3:
            out.append(w)
    return out


def split_sentences(text):
    """تقسيم جمل عربي/إنجليزي — حدود النقطة وعلامات الاستفهام والتعجب."""
    parts = re.split(r"(?<=[\.!؟\?])\s+|(?<=[\u0600-\u06FF])\n", (text or ""))
    return [p.strip() for p in parts if p and len(p.strip()) >= 25]


def score_sentence(sent, kws):
    s = sent.lower()
    hit = sum(1 for w in kws if w in s)
    return hit


def _uddg_decode(url):
    """فك روابط DuckDuckGo redirect لو ظهرت (uddg=...)."""
    m = re.search(r"uddg=([^&]+)", url)
    return unquote(m.group(1)) if m else url


def _bing_decode(url):
    """فك روابط Bing News apiclick.aspx: الرابط الأصلي مشفر داخل المعامل url=.
    متجرب فعليًا: apiclick.aspx?...&url=https%3a%2f%2f... → الرابط الأصلي."""
    u = unescape(url)
    if "apiclick.aspx" in u:
        m = re.search(r"[?&]url=([^&]+)", u)
        if m:
            orig = unquote(m.group(1))
            if orig.startswith("http"):
                return orig
    return u


def analyze(query, extra_urls=None, max_sites=_MAX_SITES):
    """البحث + القراءة من مواقع متعددة + التحليل + رد مفصل."""
    extra_urls = extra_urls or []
    t0 = time.time()
    kws = keywords_of(query)
    search_results, search_notes = [], []
    if not extra_urls and query:
        search_results, search_notes = search_web(query, max_results=max_sites + 6)

    # المواقع المقررة: روابط المستخدم أولاً، ثم نتائج البحث المباشرة (Bing المفكوكة)
    # بنطاقات مختلفة — وروابط Google تحويل JS فبتقرا معلنة وغالبًا بتفشل بصدق
    plan = []
    seen_dom = set()
    for u in extra_urls:
        plan.append({"url": u, "from": "رابط من رسالتك", "title": "", "snippet": ""})
        seen_dom.add(site_name(u))
    for r in search_results:
        if not r.get("direct"):
            continue                # روابط Google redirect — للملخص فقط
        dom = site_name(r["url"])
        if dom in seen_dom:
            continue                # تنويع المواقع — أكثر من مكان
        if len(plan) >= max_sites:
            break
        plan.append({"url": r["url"], "from": "نتيجة بحث", "title": r["title"],
                     "snippet": r["snippet"], "date": r.get("date", ""), "source": r.get("source", "")})
        seen_dom.add(dom)
    for r in search_results:
        if len(plan) >= max_sites:
            break
        if r.get("direct") or site_name(r["url"]) in seen_dom:
            continue
        plan.append({"url": r["url"], "from": "نتيجة بحث (تحويل JS)", "title": r["title"],
                     "snippet": r["snippet"], "date": r.get("date", ""), "source": r.get("source", "")})
        seen_dom.add(site_name(r["url"]))

    # القراءة الفعلية من كل موقع
    reads = []
    for p in plan:
        u = _uddg_decode(p["url"])
        res = fetch_page(u)
        res.update({"title": p.get("title", ""), "snippet": p.get("snippet", ""),
                    "from": p["from"], "date": p.get("date", ""), "source": p.get("source", "")})
        reads.append(res)
    ok_reads = [r for r in reads if r["ok"]]

    # التحليل: أفضل الجمل من كل موقع بتداخل الكلمات المفتاحية
    per_site = []
    total_sents = 0
    for r in ok_reads:
        sents = split_sentences(r["text"])
        total_sents += len(sents)
        ranked = sorted(sents, key=lambda s: score_sentence(s, kws), reverse=True) if kws else sents
        top = [s for s in ranked[:3] if (not kws or score_sentence(s, kws) > 0)][:3]
        if not top:
            top = ranked[:2]
        per_site.append({"site": site_name(r["url"]), "url": r["url"], "chars": r["chars"],
                         "seconds": r["seconds"], "top": top, "sents": len(sents)})

    # تركيب الرد المفصل
    lines = []
    n_ok = len(ok_reads)
    n_fail = len(reads) - n_ok
    lines.append("بحثت في الويب عن «%s» — دخلت فعليًا على %d موقع%s%s وقريت %d حرف من المواقع في %s ثانية."
                 % (query or "الروابط المرسلة", n_ok,
                    (" فشل %d" % n_fail) if n_fail else "",
                    (" (الروابط اللي بعتها)" if extra_urls else " (من نتائج البحث)"),
                    sum(r["chars"] for r in ok_reads), round(time.time() - t0, 1)))
    if search_notes:
        lines.append("مسار البحث: " + "، ".join(search_notes) + ".")
    if ok_reads and per_site:
        lines.append("")
        lines.append("الملخص من عناوين البحث والمصادر:")
        for i, r in enumerate(ok_reads[:3], 1):
            t = r.get("title") or ""
            sn = r.get("snippet") or ""
            if t or sn:
                lines.append("%d) %s%s" % (i, t + " — " if t else "", sn[:220]))
        lines.append("")
        lines.append("التحليل من كل موقع:")
        for ps in per_site:
            lines.append("• %s (%s) — قريت %s حرف وخلصت %d جملة، وأهم اللي مرتبط بسؤالك:"
                         % (ps["site"], ps["url"][:90], format_num(ps["chars"]), ps["sents"]))
            for s in ps["top"]:
                lines.append("  - " + s[:320])
        lines.append("")
        all_top = [s for ps in per_site for s in ps["top"]]
        if all_top:
            lines.append("الاستنتاج المجمّع: %s" % all_top[0][:300])
    else:
        lines.append("")
        lines.append("المواقع اللي جربتها ما ردتش بنص قابل للتحليل (%s)."
                     % "، ".join("%s: %s" % (site_name(r["url"]), r.get("why", "فشل")) for r in reads[:4]))
    if per_site:
        lines.append("")
        lines.append("المصادر (%d):" % len(per_site))
        for i, ps in enumerate(per_site, 1):
            lines.append("%d. %s" % (i, ps["url"]))
    thinking = ("وضع البحث في الويب: بحث RSS + قراءة %d موقع فعلًا (%s ناجح/%s فشل) + "
                "تحليل %d جمل بتداخل %d كلمة مفتاحية"
                % (len(reads), n_ok, n_fail, total_sents, len(kws)))
    return "\n".join(lines), thinking, {"query": query, "sites": len(reads), "ok": n_ok,
                                        "fail": n_fail, "sentences": total_sents,
                                        "keywords": len(kws), "seconds": round(time.time() - t0, 2)}


def format_num(n):
    """تنسيق رقم بفواصل آلاف."""
    return "{:,}".format(int(n))


# ---------- نقطة دخول الدماغ ----------
def try_web_research(message):
    """بيرجع (الرد، thinking) لو الرسالة نية بحث ويب — وإلا None.
    نفس عقد try_text_tool عشان الدماغ يدمجه بنفس النمط."""
    if not detect_intent(message):
        return None
    q, urls = extract_query(message)
    if not q and not urls:
        return ("اكتب اللي عايز أبحث عنه أو ضيف روابط المواقع، وأنا أدخل على أكتر من "
                "موقع وأحلل المعلومات وأجيبك رد مفصل بالمصادر.", "نية بحث من غير استعلام")
    reply, thinking, _ = analyze(q, extra_urls=urls)
    return reply, thinking


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('استخدام: python3 web_research.py "ابحث عن ..." [--urls url1 url2]')
        sys.exit(1)
    _q, _urls = extract_query(" ".join(sys.argv[1:]))
    rep, th, st = analyze(_q, extra_urls=_urls)
    print(rep)
    print()
    print("[إحصائيات]", st)
