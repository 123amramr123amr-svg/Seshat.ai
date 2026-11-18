"""
الأدوات المتقدمة لـ seshat
Advanced Tools for seshat
"""

import os
import json
import subprocess
import tempfile
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import re

# ========================================
# 1. تصفح الويب المتقدم (Advanced Web Browsing)
# ========================================

def browse_web(url: str, extract_tables: bool = False, follow_links: bool = False, max_links: int = 3) -> Dict[str, Any]:
    """
    تصفح صفحة ويب واستخراج المحتوى
    
    Args:
        url: رابط الصفحة
        extract_tables: استخراج الجداول
        follow_links: متابعة الروابط
        max_links: أقصى عدد روابط لمتابعتها
    
    Returns:
        dict: {
            'title': عنوان الصفحة,
            'content': المحتوى النصي,
            'tables': الجداول (إن وجدت),
            'links': الروابط,
            'error': رسالة الخطأ (إن وجدت)
        }
    """
    try:
        from bs4 import BeautifulSoup
        import requests
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # استخراج العنوان
        title = soup.title.string if soup.title else "بدون عنوان"
        
        # إزالة العناصر غير المرغوبة
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # استخراج المحتوى النصي
        content = soup.get_text(separator='\n', strip=True)
        
        # تنظيف النص
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        content = '\n'.join(lines)
        
        result = {
            'title': title,
            'content': content[:5000],  # أول 5000 حرف
            'url': url,
            'tables': [],
            'links': []
        }
        
        # استخراج الجداول
        if extract_tables:
            tables = soup.find_all('table')
            for table in tables[:5]:  # أول 5 جداول
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                if rows:
                    result['tables'].append(rows)
        
        # استخراج الروابط
        links = soup.find_all('a', href=True)
        for link in links[:20]:  # أول 20 رابط
            href = link['href']
            text = link.get_text(strip=True)
            if href and not href.startswith('#') and not href.startswith('javascript:'):
                if href.startswith('/'):
                    from urllib.parse import urljoin
                    href = urljoin(url, href)
                result['links'].append({'text': text, 'url': href})
        
        # متابعة الروابط (اختياري)
        if follow_links and result['links']:
            followed_content = []
            for link_data in result['links'][:max_links]:
                try:
                    sub_result = browse_web(link_data['url'], extract_tables=False, follow_links=False)
                    followed_content.append({
                        'url': link_data['url'],
                        'title': sub_result['title'],
                        'snippet': sub_result['content'][:500]
                    })
                except:
                    pass
            result['followed_links'] = followed_content
        
        return result
        
    except Exception as e:
        return {
            'error': f'فشل تصفح الصفحة: {str(e)}',
            'url': url
        }


# ========================================
# 2. تشغيل الأكواد (Code Execution)
# ========================================

def execute_code(code: str, language: str = 'python', timeout: int = 10) -> Dict[str, Any]:
    """
    تشغيل كود برمجي وإرجاع النتيجة
    
    Args:
        code: الكود المراد تشغيله
        language: لغة البرمجة (python فقط حالياً)
        timeout: المهلة بالثواني
    
    Returns:
        dict: {
            'output': الناتج,
            'error': الأخطاء (إن وجدت),
            'success': هل نجح التشغيل
        }
    """
    if language.lower() != 'python':
        return {
            'error': f'اللغة {language} غير مدعومة حالياً. يدعم Python فقط.',
            'success': False
        }
    
    try:
        # إنشاء ملف مؤقت
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        
        # تشغيل الكود
        result = subprocess.run(
            ['python3', temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir()
        )
        
        # حذف الملف المؤقت
        os.unlink(temp_file)
        
        return {
            'output': result.stdout,
            'error': result.stderr,
            'success': result.returncode == 0,
            'exit_code': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            'error': f'انتهت المهلة ({timeout} ثانية)',
            'success': False
        }
    except Exception as e:
        return {
            'error': f'خطأ في التشغيل: {str(e)}',
            'success': False
        }


# ========================================
# 3. ذاكرة طويلة المدى (Long-term Memory)
# ========================================

class LongTermMemory:
    """ذاكرة طويلة المدى لـ seshat"""
    
    def __init__(self, memory_file: str = 'data/long_memory.json'):
        self.memory_file = Path(memory_file)
        self.memories = self._load()
    
    def _load(self) -> List[Dict]:
        """تحميل الذكريات من الملف"""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save(self):
        """حفظ الذكريات في الملف"""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)
    
    def remember(self, content: str, tags: List[str] = None, importance: int = 5):
        """
        حفظ ذكرى جديدة
        
        Args:
            content: محتوى الذكرى
            tags: وسوم للبحث
            importance: الأهمية (1-10)
        """
        memory = {
            'content': content,
            'tags': tags or [],
            'importance': importance,
            'timestamp': datetime.now().isoformat(),
            'id': hashlib.md5(content.encode()).hexdigest()[:8]
        }
        self.memories.append(memory)
        self._save()
        return memory
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        """
        استرجاع ذكريات متعلقة بالاستعلام
        
        Args:
            query: نص البحث
            limit: عدد النتائج
        
        Returns:
            list: الذكريات المتعلقة
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        for memory in self.memories:
            content_lower = memory['content'].lower()
            
            # حساب النقاط
            score = 0
            
            # تطابق الكلمات
            content_words = set(content_lower.split())
            common_words = query_words & content_words
            score += len(common_words) * 2
            
            # تطابق الوسوم
            for tag in memory.get('tags', []):
                if tag.lower() in query_lower:
                    score += 3
            
            # الأهمية
            score += memory.get('importance', 5) / 10
            
            if score > 0:
                scored.append((score, memory))
        
        # ترتيب حسب النقاط
        scored.sort(reverse=True, key=lambda x: x[0])
        
        return [mem for _, mem in scored[:limit]]
    
    def forget(self, memory_id: str) -> bool:
        """نسيان ذكرى معينة"""
        for i, memory in enumerate(self.memories):
            if memory.get('id') == memory_id:
                self.memories.pop(i)
                self._save()
                return True
        return False
    
    def get_all(self, limit: int = 100) -> List[Dict]:
        """جلب كل الذكريات"""
        return self.memories[-limit:]


# ========================================
# 4. إنشاء الملفات (File Generation)
# ========================================

def create_text_file(content: str, filename: str = None) -> Dict[str, Any]:
    """
    إنشاء ملف نصي
    
    Args:
        content: محتوى الملف
        filename: اسم الملف (اختياري)
    
    Returns:
        dict: {'path': مسار الملف, 'success': هل نجح}
    """
    try:
        # إنشاء مجلد الملفات
        files_dir = Path('data/files')
        files_dir.mkdir(parents=True, exist_ok=True)
        
        # تحديد اسم الملف
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'file_{timestamp}.txt'
        
        # إضافة .txt إن لم يكن موجوداً
        if not filename.endswith('.txt'):
            filename += '.txt'
        
        filepath = files_dir / filename
        
        # كتابة الملف
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            'path': str(filepath),
            'filename': filename,
            'size': len(content),
            'success': True
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


def create_pdf_file(content: str, title: str = "مستند", filename: str = None) -> Dict[str, Any]:
    """
    إنشاء ملف PDF
    
    Args:
        content: محتوى الملف
        title: عنوان المستند
        filename: اسم الملف (اختياري)
    
    Returns:
        dict: {'path': مسار الملف, 'success': هل نجح}
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.units import cm
        
        # إنشاء مجلد الملفات
        files_dir = Path('data/files')
        files_dir.mkdir(parents=True, exist_ok=True)
        
        # تحديد اسم الملف
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'document_{timestamp}.pdf'
        
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        filepath = files_dir / filename
        
        # إنشاء PDF
        c = canvas.Canvas(str(filepath), pagesize=A4)
        width, height = A4
        
        # محاولة استخدام خط عربي
        arabic_font = 'Helvetica'  # fallback
        try:
            # محاولة تسجيل خط عربي
            if os.path.exists('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
                pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
                arabic_font = 'DejaVu'
        except:
            pass
        
        # العنوان
        c.setFont(arabic_font, 16)
        c.drawString(2*cm, height - 2*cm, title)
        
        # المحتوى
        c.setFont(arabic_font, 12)
        y_position = height - 4*cm
        
        lines = content.split('\n')
        for line in lines:
            if y_position < 2*cm:  # صفحة جديدة
                c.showPage()
                c.setFont(arabic_font, 12)
                y_position = height - 2*cm
            
            # تقسيم الأسطر الطويلة
            words = line.split()
            current_line = ''
            for word in words:
                test_line = current_line + ' ' + word if current_line else word
                if len(test_line) < 80:  # حد أقصى للحروف
                    current_line = test_line
                else:
                    c.drawString(2*cm, y_position, current_line)
                    y_position -= 0.5*cm
                    current_line = word
            
            if current_line:
                c.drawString(2*cm, y_position, current_line)
                y_position -= 0.5*cm
            
            y_position -= 0.3*cm
        
        c.save()
        
        return {
            'path': str(filepath),
            'filename': filename,
            'title': title,
            'success': True
        }
        
    except ImportError:
        return {
            'error': 'مكتبة reportlab غير مثبتة. ثبتها بـ: pip install reportlab',
            'success': False
        }
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


# ========================================
# 5. فهم الصور (Image Understanding)
# ========================================

def understand_image(image_path: str) -> Dict[str, Any]:
    """
    فهم محتوى صورة (OCR + وصف أساسي)
    
    Args:
        image_path: مسار الصورة
    
    Returns:
        dict: {
            'text': النصوص المستخرجة,
            'description': وصف أساسي,
            'success': هل نجح
        }
    """
    try:
        from PIL import Image
        
        # فتح الصورة
        img = Image.open(image_path)
        
        result = {
            'path': image_path,
            'size': img.size,
            'format': img.format,
            'mode': img.mode,
            'text': '',
            'description': '',
            'success': True
        }
        
        # وصف أساسي
        width, height = img.size
        result['description'] = f"صورة بأبعاد {width}×{height} بكسل، صيغة {img.format}"
        
        # محاولة OCR
        try:
            import pytesseract
            
            # تحويل لأبيض وأسود للدقة
            if img.mode != 'L':
                img_gray = img.convert('L')
            else:
                img_gray = img
            
            # استخراج النص
            text = pytesseract.image_to_string(img_gray, lang='ara+eng')
            text = text.strip()
            
            if text:
                result['text'] = text
                result['description'] += f"، تحتوي على نص: {text[:100]}..."
        
        except ImportError:
            result['text'] = 'مكتبة pytesseract غير مثبتة'
        except Exception as e:
            result['text'] = f'فشل استخراج النص: {str(e)}'
        
        return result
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


# ========================================
# واجهة موحدة للأدوات (Unified Tools Interface)
# ========================================

class SeshatTools:
    """واجهة موحدة لكل أدوات seshat"""
    
    def __init__(self):
        self.memory = LongTermMemory()
    
    def browse(self, url: str, **kwargs) -> Dict:
        """تصفح الويب"""
        return browse_web(url, **kwargs)
    
    def execute(self, code: str, language: str = 'python') -> Dict:
        """تشغيل كود"""
        return execute_code(code, language)
    
    def remember(self, content: str, **kwargs) -> Dict:
        """حفظ ذكرى"""
        return self.memory.remember(content, **kwargs)
    
    def recall(self, query: str, **kwargs) -> List:
        """استرجاع ذكريات"""
        return self.memory.recall(query, **kwargs)
    
    def create_file(self, content: str, file_type: str = 'text', **kwargs) -> Dict:
        """إنشاء ملف"""
        if file_type.lower() == 'pdf':
            return create_pdf_file(content, **kwargs)
        else:
            return create_text_file(content, **kwargs)
    
    def see_image(self, image_path: str) -> Dict:
        """فهم صورة"""
        return understand_image(image_path)


# إنشاء نسخة عامة
tools = SeshatTools()
