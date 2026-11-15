"""
Token Analyzer for Arabic Text
تحليل الكلمات العربية: تطبيع، جذر صرفي، تصنيف نحوي، وزن دلالي

Features:
- Arabic normalization (remove diacritics, normalize alef/ya/taa)
- Root stem extraction (Arabic triliteral roots)
- POS tagging (noun, verb, particle, interrogative, tech term)
- IDF weight calculation
- Semantic weight based on position and type
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnalyzedToken:
    """ناتج تحليل كلمة واحدة"""
    token: str              # الكلمة الأصلية
    normalized: str         # بعد التطبيع
    stem: str              # الجذر الصرفي
    pos: str               # النوع (NOUN, VERB, PARTICLE, INTERROGATIVE, TECH_TERM)
    pos_label_ar: str      # التصنيف بالعربي
    semantic_weight: float # الوزن الدلالي (0-1)
    idf_weight: float      # وزن IDF
    frequency: int         # عدد مرات الظهور في النص


class ArabicNormalizer:
    """تطبيع النص العربي"""
    
    # خريطة التطبيع
    NORMALIZATION_MAP = {
        'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',  # أنواع الألف
        'ى': 'ي',  # الألف المقصورة
        'ة': 'ه',  # التاء المربوطة
        'ؤ': 'و',  # الواو مع همزة
        'ئ': 'ي',  # الياء مع همزة
    }
    
    # التشكيل والحركات
    DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]')
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """
        تطبيع النص العربي:
        1. إزالة التشكيل
        2. توحيد أنواع الألف
        3. توحيد التاء المربوطة
        4. توحيد الياء والألف المقصورة
        """
        # إزالة التشكيل
        text = cls.DIACRITICS.sub('', text)
        
        # توحيد الحروف
        for old, new in cls.NORMALIZATION_MAP.items():
            text = text.replace(old, new)
        
        return text


class RootExtractor:
    """استخراج الجذر الصرفي للكلمات العربية"""
    
    # الجذور الشائعة (ثلاثية)
    COMMON_ROOTS = {
        'كتب': ['كاتب', 'مكتوب', 'كتاب', 'مكتبة', 'يكتب', 'كتب'],
        'قرا': ['قارئ', 'مقروء', 'قرآن', 'قراءة', 'يقرأ', 'قرأ'],
        'علم': ['عالم', 'معلوم', 'علم', 'معلم', 'يعلم', 'علماء'],
        'فهم': ['فاهم', 'مفهوم', 'فهم', 'يفهم'],
        'سأل': ['سائل', 'مسؤول', 'سؤال', 'يسأل'],
        'رد': ['راد', 'مردود', 'رد', 'يرد'],
        'عمل': ['عامل', 'معمول', 'عمل', 'يعمل'],
        'قول': ['قائل', 'مقول', 'قول', 'يقول'],
        'بنى': ['باني', 'مبني', 'بناء', 'يبني'],
        'درس': ['دارس', 'مدروس', 'درس', 'يدرس'],
    }
    
    # السوابق الشائعة
    PREFIXES = ['ال', 'وال', 'فال', 'بال', 'كال', 'لل', 'ف', 'و', 'ب', 'ك', 'ل']
    
    # اللواحق الشائعة
    SUFFIXES = ['ون', 'ين', 'ات', 'ة', 'ها', 'هم', 'هن', 'نا', 'كم', 'كن']
    
    @classmethod
    def extract_root(cls, word: str) -> str:
        """
        استخراج الجذر الصرفي (تجريبي):
        1. إزالة السوابق واللواحق
        2. البحث في الجذور الشائعة
        3. استخراج الحروف الأصلية (3-4 حروف)
        """
        normalized = ArabicNormalizer.normalize(word)
        
        # إزالة السوابق
        for prefix in sorted(cls.PREFIXES, key=len, reverse=True):
            if normalized.startswith(prefix) and len(normalized) > len(prefix) + 2:
                normalized = normalized[len(prefix):]
                break
        
        # إزالة اللواحق
        for suffix in sorted(cls.SUFFIXES, key=len, reverse=True):
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 2:
                normalized = normalized[:-len(suffix)]
                break
        
        # البحث في الجذور الشائعة
        for root, derivatives in cls.COMMON_ROOTS.items():
            if normalized in derivatives or root in normalized:
                return root
        
        # استخراج أول 3 حروف (جذر تجريبي)
        if len(normalized) >= 3:
            return normalized[:3]
        
        return normalized


class POSTagger:
    """تصنيف الكلمات نحويًا"""
    
    # أدوات الاستفهام
    INTERROGATIVES = {
        'ما', 'ماذا', 'من', 'متى', 'أين', 'كيف', 'لماذا', 'كم', 'أي',
        'هل', 'أ', 'اين', 'متي', 'ليه', 'ازاي', 'ايه', 'مين'
    }
    
    # الأفعال الشائعة
    COMMON_VERBS = {
        'كان', 'يكون', 'عمل', 'يعمل', 'قال', 'يقول', 'فعل', 'يفعل',
        'ذهب', 'يذهب', 'كتب', 'يكتب', 'قرأ', 'يقرأ', 'فهم', 'يفهم',
        'سأل', 'يسأل', 'رد', 'يرد', 'بنى', 'يبني', 'درس', 'يدرس'
    }
    
    # حروف الجر والعطف
    PARTICLES = {
        'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بين', 'قبل', 'بعد',
        'و', 'أو', 'ثم', 'لكن', 'بل', 'لا', 'ما', 'لم', 'لن', 'قد'
    }
    
    # مصطلحات تقنية
    TECH_TERMS = {
        'ذكاء', 'اصطناعي', 'نموذج', 'شبكة', 'عصبية', 'تعلم', 'آلي',
        'خوارزمية', 'بيانات', 'تدريب', 'انتباه', 'متجه', 'دلالة',
        'ذكاء', 'اصطناعي', 'نموذج', 'شبكة', 'عصبية', 'تعلم', 'آلي',
        'ai', 'model', 'neural', 'network', 'attention', 'embedding'
    }
    
    @classmethod
    def tag(cls, word: str) -> Tuple[str, str]:
        """
        تصنيف الكلمة نحويًا
        Returns: (POS tag, Arabic label)
        """
        normalized = ArabicNormalizer.normalize(word.lower())
        
        # أدوات الاستفهام
        if normalized in cls.INTERROGATIVES:
            return 'INTERROGATIVE', 'استفهام'
        
        # مصطلحات تقنية
        if normalized in cls.TECH_TERMS:
            return 'TECH_TERM', 'مصطلح تقني'
        
        # أفعال
        if normalized in cls.COMMON_VERBS:
            return 'VERB', 'فعل'
        
        # حروف
        if normalized in cls.PARTICLES:
            return 'PARTICLE', 'حرف'
        
        # افتراضي: اسم
        return 'NOUN', 'اسم'


class IDFCalculator:
    """حساب وزن IDF (Inverse Document Frequency)"""
    
    @staticmethod
    def compute_idf(token: str, doc_frequency: int, total_docs: int) -> float:
        """
        حساب IDF:
        IDF(t) = log((N + 1) / (df(t) + 1)) + 1
        
        حيث:
        - N: العدد الكلي للوثائق
        - df(t): عدد الوثائق اللي ظهر فيها الكلمة t
        """
        if doc_frequency == 0:
            doc_frequency = 1
        
        idf = math.log((total_docs + 1) / (doc_frequency + 1)) + 1
        return idf


class TokenAnalyzer:
    """محلل الكلمات الرئيسي"""
    
    def __init__(self, corpus: Optional[List[str]] = None):
        """
        تهيئة المحلل
        
        Args:
            corpus: قائمة النصوص لحساب IDF
        """
        self.corpus = corpus or []
        self.doc_frequencies = self._compute_doc_frequencies()
        self.total_docs = len(self.corpus) if self.corpus else 1000  # افتراضي
    
    def _compute_doc_frequencies(self) -> Dict[str, int]:
        """حساب تكرار الكلمات في الوثائق"""
        doc_freq = {}
        
        for doc in self.corpus:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                normalized = ArabicNormalizer.normalize(token)
                doc_freq[normalized] = doc_freq.get(normalized, 0) + 1
        
        return doc_freq
    
    def _tokenize(self, text: str) -> List[str]:
        """تقسيم النص إلى كلمات"""
        # إزالة علامات الترقيم
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def analyze(self, text: str) -> List[AnalyzedToken]:
        """
        تحليل نص كامل
        
        Args:
            text: النص المراد تحليله
        
        Returns:
            قائمة AnalyzedToken لكل كلمة
        """
        tokens = self._tokenize(text)
        token_counts = {}
        
        # حساب التكرار
        for token in tokens:
            normalized = ArabicNormalizer.normalize(token)
            token_counts[normalized] = token_counts.get(normalized, 0) + 1
        
        analyzed = []
        for token in tokens:
            normalized = ArabicNormalizer.normalize(token)
            
            # استخراج الجذر
            stem = RootExtractor.extract_root(token)
            
            # التصنيف النحوي
            pos, pos_label = POSTagger.tag(token)
            
            # حساب IDF
            doc_freq = self.doc_frequencies.get(normalized, 1)
            idf = IDFCalculator.compute_idf(normalized, doc_freq, self.total_docs)
            
            # حساب الوزن الدلالي
            semantic_weight = self._compute_semantic_weight(
                token, pos, idf, token_counts[normalized]
            )
            
            analyzed.append(AnalyzedToken(
                token=token,
                normalized=normalized,
                stem=stem,
                pos=pos,
                pos_label_ar=pos_label,
                semantic_weight=semantic_weight,
                idf_weight=idf,
                frequency=token_counts[normalized]
            ))
        
        return analyzed
    
    def _compute_semantic_weight(
        self,
        token: str,
        pos: str,
        idf: float,
        frequency: int
    ) -> float:
        """
        حساب الوزن الدلالي للكلمة
        
        العوامل:
        - نوع الكلمة (أدوات الاستفهام أعلى وزن)
        - IDF (الكلمات النادرة أعلى وزن)
        - التكرار (التكرار العالي يقلل الوزن)
        - طول الكلمة (الكلمات الأطول غالبًا أكثر دلالة)
        """
        weight = 0.5  # وزن أساسي
        
        # نوع الكلمة
        if pos == 'INTERROGATIVE':
            weight += 0.3
        elif pos == 'TECH_TERM':
            weight += 0.25
        elif pos == 'VERB':
            weight += 0.15
        elif pos == 'NOUN':
            weight += 0.1
        
        # IDF (كلمات نادرة = وزن أعلى)
        if idf > 3.0:
            weight += 0.2
        elif idf > 2.0:
            weight += 0.1
        
        # التكرار (تكرار عالي = وزن أقل)
        if frequency > 3:
            weight -= 0.1
        elif frequency == 1:
            weight += 0.1
        
        # طول الكلمة
        if len(token) > 5:
            weight += 0.1
        
        # تحديد بين 0 و 1
        return max(0.0, min(1.0, weight))


# Convenience function
def analyze_text(text: str, corpus: Optional[List[str]] = None) -> List[AnalyzedToken]:
    """
    دالة مساعدة لتحليل نص
    
    Args:
        text: النص المراد تحليله
        corpus: قائمة النصوص لحساب IDF (اختياري)
    
    Returns:
        قائمة AnalyzedToken
    """
    analyzer = TokenAnalyzer(corpus=corpus)
    return analyzer.analyze(text)


# Test
if __name__ == '__main__':
    # مثال على الاستخدام
    sample_corpus = [
        'الذكاء الاصطناعي يتعلم من البيانات',
        'النموذج يفهم الأسئلة ويرد عليها',
        'الشبكة العصبية تحاكي الدماغ البشري',
    ]
    
    test_text = 'ازاي النموذج ده بيفهم الأسئلة ويعمل ردود جديدة؟'
    
    analyzer = TokenAnalyzer(corpus=sample_corpus)
    tokens = analyzer.analyze(test_text)
    
    print(f"تحليل: {test_text}\n")
    print(f"{'الكلمة':<15} {'المطبع':<15} {'الجذر':<10} {'النوع':<15} {'الوزن':<8} {'IDF':<8}")
    print('-' * 80)
    
    for token in tokens:
        print(
            f"{token.token:<15} "
            f"{token.normalized:<15} "
            f"{token.stem:<10} "
            f"{token.pos_label_ar:<15} "
            f"{token.semantic_weight:<8.3f} "
            f"{token.idf_weight:<8.3f}"
        )
