# -*- coding: utf-8 -*-
"""استبدال قسم خوفو المدمج القديم + JS التبويبات في templates/index.html."""
import io
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates", "index.html")
src = io.open(PATH, encoding="utf-8").read()

# ---------- 1) قسم خوفو: من <div id="khufu-section" إلى قبل <footer> ----------
start = src.index('<div id="khufu-section"')
end = src.index("<footer>")
new_section = (
    '<div id="khufu-section" style="display:none; position:absolute; top:0; left:0; right:0; '
    'z-index:100; background:var(--bg);">\n'
    '  <iframe id="khufu-frame" title="خوفو — مولّد الصور" '
    'style="width:100%; height:100%; border:0; display:block;\"></iframe>\n'
    '</div>\n\n'
)
src = src[:start] + new_section + src[end:]

# ---------- 2) JS التبديل: من تعليق التبديل إلى آخر السكربت ----------
jstart = src.index("  // ---------------- التبديل بين النماذج ----------------")
jend = src.index("</script>", jstart)
new_js = """  // ---------------- التبديل بين النماذج ----------------
  // خوفو بيتحمّل في iframe من /khufu-app (المصدر الوحيد لواجهته) — أول مرة تتفتح بس.
  var seshatMain = document.getElementById("scroll");
  var seshatFooter = document.querySelector("footer");
  var khufuSection = document.getElementById("khufu-section");
  var khufuFrame = document.getElementById("khufu-frame");

  document.querySelectorAll(".model-tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      document.querySelectorAll(".model-tab").forEach(function (t) { t.classList.remove("active"); });
      tab.classList.add("active");
      var model = tab.dataset.model;
      if (model === "seshat") {
        seshatMain.style.display = "";
        seshatFooter.style.display = "";
        khufuSection.style.display = "none";
      } else if (model === "khufu") {
        var hdr = document.querySelector("header");
        seshatMain.style.display = "none";
        seshatFooter.style.display = "none";
        khufuSection.style.top = hdr.offsetHeight + "px";
        khufuSection.style.height = "calc(100vh - " + hdr.offsetHeight + "px)";
        khufuSection.style.display = "block";
        if (!khufuFrame.src) khufuFrame.src = "/khufu-app";
      }
    });
  });

  input.focus();
"""
src = src[:jstart] + new_js + src[jend:]

io.open(PATH, "w", encoding="utf-8").write(src)

# تحقق سريع
chk = io.open(PATH, encoding="utf-8").read()
assert 'khufu-frame" title' in chk, "iframe مش موجود"
assert "khufu-1.0" not in chk, "النسخة القديمة لسه موجودة"
assert "loadKhufuStatus" not in chk, "الـ JS القديم لسه موجود"
assert "المصدر الوحيد لواجهته" in chk, "الـ JS الجديد مش متزرع"
assert chk.count("khufu-section") == 2, "عدد مرات ذكر khufu-section غلط"
print("تم الاستبدال بنجاح — الطول:", len(chk.splitlines()), "سطر")
