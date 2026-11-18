"""
دمج الأدوات المتقدمة مع seshat brain
Integration of Advanced Tools with seshat brain
"""

import re
import os
from typing import Optional, Dict, Any
from tools import tools, browse_web, execute_code, create_text_file, create_pdf_file, understand_image

def detect_tool_request(message: str) -> Optional[Dict[str, Any]]:
    """
    اكتشاف طلب أداة من رسالة المستخدم
    
    Returns:
        dict: {'tool': اسم الأداة, 'params': المعاملات} أو None
    """
    msg_lower = message.lower().strip()
    
    # 1. تصفح الويب المتقدم
    browse_patterns = [
        r'(تصفح|افتح|ادخل على|زور) (.+)',
        r'(browse|open|visit) (.+)',
        r'(اقرأ|اقرا) (الصفحة|الموقع) (.+)',
    ]
    
    for pattern in browse_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            url = match.group(2).strip()
            if not url.startswith('http'):
                url = 'https://' + url
            return {
                'tool': 'browse',
                'params': {'url': url, 'extract_tables': True}
            }
    
    # 2. تشغيل الأكواد
    code_patterns = [
        r'(شغل|نفذ|execute|run) (الكود|code):\s*```(.+?)```',
        r'(احسب|calculate) (.+)',
    ]
    
    for pattern in code_patterns:
        match = re.search(pattern, msg_lower, re.DOTALL)
        if match:
            if 'احسب' in msg_lower or 'calculate' in msg_lower:
                # تحويل لعملية حسابية
                expr = match.group(2).strip()
                code = f"print({expr})"
            else:
                code = match.group(3).strip()
            
            return {
                'tool': 'execute',
                'params': {'code': code, 'language': 'python'}
            }
    
    # 3. إنشاء ملفات
    file_patterns = [
        r'(انشئ|اعمل|create) (ملف|file) (.+)',
        r'(احفظ|save) (في|to) (ملف|file) (.+)',
    ]
    
    for pattern in file_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            # استخراج المحتوى من السياق
            return {
                'tool': 'create_file',
                'params': {'content': message, 'file_type': 'text'}
            }
    
    # 4. PDF
    pdf_patterns = [
        r'(انشئ|اعمل|create) (pdf|بي دي اف) (.+)',
        r'(احفظ|save) (pdf|بي دي اف) (.+)',
    ]
    
    for pattern in pdf_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            return {
                'tool': 'create_pdf',
                'params': {'content': message, 'title': match.group(3)}
            }
    
    # 5. فهم الصور
    image_patterns = [
        r'(شوف|فهم|analyze) (الصورة|image) (.+)',
        r'(اقرا|read) (النص|text) (في|from) (الصورة|image) (.+)',
    ]
    
    for pattern in image_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            image_path = match.group(3).strip()
            return {
                'tool': 'understand_image',
                'params': {'image_path': image_path}
            }
    
    # 6. الذاكرة
    memory_patterns = [
        r'(اتذكر|remember) (.+)',
        r'(افتكر|recall) (.+)',
    ]
    
    if re.search(r'(اتذكر|remember)', msg_lower):
        content = message.split('اتذكر', 1)[-1].split('remember', 1)[-1].strip()
        return {
            'tool': 'remember',
            'params': {'content': content, 'importance': 7}
        }
    
    if re.search(r'(افتكر|recall)', msg_lower):
        query = message.split('افتكر', 1)[-1].split('recall', 1)[-1].strip()
        return {
            'tool': 'recall',
            'params': {'query': query}
        }
    
    return None


def execute_tool_request(tool_request: Dict[str, Any]) -> str:
    """
    تنفيذ طلب أداة وإرجاع الرد
    
    Returns:
        str: الرد المنسق
    """
    tool = tool_request['tool']
    params = tool_request['params']
    
    try:
        if tool == 'browse':
            result = browse_web(**params)
            if 'error' in result:
                return f"❌ {result['error']}"
            
            response = f"🌐 **{result['title']}**\n\n"
            response += f"📄 المحتوى:\n{result['content'][:2000]}\n\n"
            
            if result.get('tables'):
                response += f"📊 الجداول: {len(result['tables'])} جدول\n"
            
            if result.get('links'):
                response += f"🔗 الروابط: {len(result['links'])} رابط\n"
                for link in result['links'][:5]:
                    response += f"  • {link['text']}: {link['url']}\n"
            
            return response
        
        elif tool == 'execute':
            result = execute_code(**params)
            if result['success']:
                return f"✅ **تم التشغيل بنجاح**\n\n```\n{result['output']}\n```"
            else:
                return f"❌ **خطأ في التشغيل**\n\n```\n{result['error']}\n```"
        
        elif tool == 'create_file':
            result = create_text_file(**params)
            if result['success']:
                return f"✅ **تم إنشاء الملف**\n\n📁 المسار: {result['path']}\n📊 الحجم: {result['size']} حرف"
            else:
                return f"❌ فشل إنشاء الملف: {result['error']}"
        
        elif tool == 'create_pdf':
            result = create_pdf_file(**params)
            if result['success']:
                return f"✅ **تم إنشاء PDF**\n\n📁 المسار: {result['path']}\n📄 العنوان: {result['title']}"
            else:
                return f"❌ فشل إنشاء PDF: {result['error']}"
        
        elif tool == 'understand_image':
            result = understand_image(**params)
            if result['success']:
                response = f"🖼️ **تحليل الصورة**\n\n"
                response += f"📏 الأبعاد: {result['size']}\n"
                response += f"📝 الوصف: {result['description']}\n"
                if result.get('text'):
                    response += f"\n📄 النص المستخرج:\n```\n{result['text']}\n```"
                return response
            else:
                return f"❌ فشل تحليل الصورة: {result['error']}"
        
        elif tool == 'remember':
            memory = tools.remember(**params)
            return f"✅ **تم الحفظ في الذاكرة**\n\n💾 المعرف: {memory['id']}\n📝 المحتوى: {memory['content'][:100]}..."
        
        elif tool == 'recall':
            memories = tools.recall(**params)
            if memories:
                response = f"🧠 **الذكريات المتعلقة** ({len(memories)} ذكرى):\n\n"
                for i, mem in enumerate(memories, 1):
                    response += f"{i}. {mem['content'][:150]}...\n"
                return response
            else:
                return "🤔 مفيش ذكريات متعلقة بالموضوع ده"
        
        else:
            return f"❌ أداة غير معروفة: {tool}"
    
    except Exception as e:
        return f"❌ خطأ في تنفيذ الأداة: {str(e)}"


def process_with_tools(message: str, default_reply_func) -> str:
    """
    معالجة رسالة مع الأدوات المتقدمة
    
    Args:
        message: رسالة المستخدم
        default_reply_func: دالة الرد الافتراضية (من brain)
    
    Returns:
        str: الرد
    """
    # اكتشاف طلب أداة
    tool_request = detect_tool_request(message)
    
    if tool_request:
        # تنفيذ الأداة
        return execute_tool_request(tool_request)
    
    # مفيش أداة - استخدم الرد الافتراضي
    return default_reply_func(message)
