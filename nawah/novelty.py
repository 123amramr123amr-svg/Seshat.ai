"""
Novelty Score Calculator for Nawah
قياس أصالة الردود: كم الرد مبتكر مش منسوخ

Features:
- Compare response with corpus
- Calculate novelty score (0-1)
- Identify copied phrases
- Support for n-gram similarity
- Cosine similarity for semantic comparison
"""

import math
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class NoveltyResult:
    """نتيجة قياس الأصالة"""
    novelty_score: float          # درجة الأصالة (0-1)
    is_novel: bool                # هل الرد مبتكر؟
    copied_phrases: List[str]     # العبارات المنسوخة
    similarity_breakdown: Dict[str, float]  # تفاصيل التشابه
    matched_corpus_ids: List[int] # معرفات النصوص المطابقة
    
    def to_dict(self) -> Dict:
        return {
            'novelty_score': self.novelty_score,
            'is_novel': self.is_novel,
            'copied_phrases': self.copied_phrases,
            'similarity_breakdown': self.similarity_breakdown,
            'matched_corpus_ids': self.matched_corpus_ids
        }


class NGramExtractor:
    """استخراج N-grams من النص"""
    
    @staticmethod
    def extract(text: str, n: int = 2) -> List[str]:
        """
        استخراج N-grams (تتابع n كلمة)
        
        Args:
            text: النص
            n: عدد الكلمات في كل gram
        
        Returns:
            قائمة N-grams
        """
        words = text.split()
        if len(words) < n:
            return [text]
        
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        
        return ngrams


class CosineSimilarity:
    """حساب تشابه Cosine بين متجهين"""
    
    @staticmethod
    def compute(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """
        حساب Cosine Similarity
        
        Args:
            vec1: المتجه الأول (كلمة -> تكرار)
            vec2: المتجه الثاني (كلمة -> تكرار)
        
        Returns:
            درجة التشابه (0-1)
        """
        # جلب كل الكلمات الفريدة
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        # حساب الضرب النقطي
        dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)
        
        # حساب الأقطار
        magnitude1 = math.sqrt(sum(v**2 for v in vec1.values()))
        magnitude2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class NoveltyCalculator:
    """حاسب الأصالة"""
    
    def __init__(self, corpus: Optional[List[Dict]] = None):
        """
        Args:
            corpus: قائمة النصوص المرجعية
                    [{'id': 1, 'text': '...'}, ...]
        """
        self.corpus = corpus or []
        self.ngram_extractor = NGramExtractor()
    
    def calculate(
        self,
        response: str,
        question: str = '',
        threshold: float = 0.7
    ) -> NoveltyResult:
        """
        حساب درجة الأصالة للرد
        
        Args:
            response: الرد المراد تقييمه
            question: السؤال الأصلي (لحساب التشابه مع السؤال)
            threshold: حد التشابه (أعلى منه = منسوخ)
        
        Returns:
            NoveltyResult
        """
        similarities = {}
        copied_phrases = []
        matched_ids = []
        
        # 1. التشابه مع السؤال (ردود منسوخة من السؤال = غير مبتكرة)
        if question:
            q_similarity = self._compute_similarity(response, question)
            similarities['question_similarity'] = q_similarity
            
            # استخراج العبارات المشتركة
            common = self._find_common_phrases(response, question)
            copied_phrases.extend(common)
        
        # 2. التشابه مع النصوص في Corpus
        for doc in self.corpus:
            doc_text = doc.get('text', '')
            doc_id = doc.get('id', 0)
            
            similarity = self._compute_similarity(response, doc_text)
            similarities[f'corpus_{doc_id}'] = similarity
            
            if similarity > threshold:
                matched_ids.append(doc_id)
                
                # استخراج العبارات المنسوخة
                common = self._find_common_phrases(response, doc_text)
                copied_phrases.extend(common)
        
        # 3. حساب درجة الأصالة
        max_similarity = max(similarities.values()) if similarities else 0.0
        
        # الأصالة = 1 - أقصى تشابه
        novelty_score = max(0.0, 1.0 - max_similarity)
        
        # 4. مكافأة الردود الطويلة (عادةً أكثر تفصيلاً)
        word_count = len(response.split())
        if word_count > 20:
            novelty_score = min(1.0, novelty_score + 0.1)
        
        # 5. مكافأة الردود المتنوعة (كلمات فريدة كثيرة)
        unique_words = len(set(response.split()))
        diversity = unique_words / max(word_count, 1)
        if diversity > 0.7:
            novelty_score = min(1.0, novelty_score + 0.05)
        
        # إزالة التكرار من العبارات المنسوخة
        copied_phrases = list(set(copied_phrases))
        
        # هل الرد مبتكر؟
        is_novel = novelty_score > 0.5
        
        return NoveltyResult(
            novelty_score=novelty_score,
            is_novel=is_novel,
            copied_phrases=copied_phrases,
            similarity_breakdown=similarities,
            matched_corpus_ids=matched_ids
        )
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """حساب التشابه بين نصين"""
        # استخراج الكلمات
        words1 = Counter(text1.split())
        words2 = Counter(text2.split())
        
        # Cosine similarity على الكلمات
        cosine_sim = CosineSimilarity.compute(words1, words2)
        
        # N-gram similarity (bigrams)
        bigrams1 = self.ngram_extractor.extract(text1, n=2)
        bigrams2 = self.ngram_extractor.extract(text2, n=2)
        
        if bigrams1 and bigrams2:
            bigram_counter1 = Counter(bigrams1)
            bigram_counter2 = Counter(bigrams2)
            bigram_sim = CosineSimilarity.compute(bigram_counter1, bigram_counter2)
        else:
            bigram_sim = 0.0
        
        # متوسط التشابهين
        return (cosine_sim + bigram_sim) / 2
    
    def _find_common_phrases(self, text1: str, text2: str, min_length: int = 3) -> List[str]:
        """
        إيجاد العبارات المشتركة بين نصين
        
        Args:
            text1: النص الأول
            text2: النص الثاني
            min_length: أقل عدد كلمات للعبرة المشتركة
        
        Returns:
            قائمة العبارات المشتركة
        """
        words1 = text1.split()
        words2 = text2.split()
        
        common_phrases = []
        
        # البحث عن تتابعات مشتركة
        for i in range(len(words1)):
            for j in range(len(words2)):
                # محاولة مطابقة تتابع
                match_length = 0
                while (i + match_length < len(words1) and
                       j + match_length < len(words2) and
                       words1[i + match_length] == words2[j + match_length]):
                    match_length += 1
                
                # إذا التطابق طويل كفاية
                if match_length >= min_length:
                    phrase = ' '.join(words1[i:i + match_length])
                    common_phrases.append(phrase)
        
        # إزالة التكرار
        return list(set(common_phrases))


class NoveltyBooster:
    """تعزيز الأصالة في توليد الردود"""
    
    @staticmethod
    def boost_response(
        response: str,
        novelty_result: NoveltyResult,
        boost_factor: float = 0.85
    ) -> str:
        """
        تعزيز رد منخفض الأصالة
        
        Args:
            response: الرد الأصلي
            novelty_result: نتيجة قياس الأصالة
            boost_factor: معامل التعزيز
        
        Returns:
            الرد المعزز
        """
        if novelty_result.is_novel:
            return response
        
        # إضافة عبارات توضيحية
        boost_phrases = [
            'بمعنى آخر',
            'بتعبير مختلف',
            'ويمكن القول أيضًا',
            'من ناحية أخرى',
            'علاوة على ذلك'
        ]
        
        # اختيار عبارة عشوائية (بسيط)
        import random
        boost = random.choice(boost_phrases)
        
        # إضافة التعزيز في البداية
        enhanced = f"{boost}، {response}"
        
        return enhanced


# Convenience function
def calculate_novelty(
    response: str,
    question: str = '',
    corpus: Optional[List[Dict]] = None
) -> NoveltyResult:
    """
    دالة مساعدة لحساب الأصالة
    
    Args:
        response: الرد المراد تقييمه
        question: السؤال الأصلي (اختياري)
        corpus: قائمة النصوص المرجعية (اختياري)
    
    Returns:
        NoveltyResult
    """
    calculator = NoveltyCalculator(corpus=corpus)
    return calculator.calculate(response, question)


# Test
if __name__ == '__main__':
    # مثال على الاستخدام
    corpus = [
        {'id': 1, 'text': 'النموذج يفهم الأسئلة ويرد عليها'},
        {'id': 2, 'text': 'الذكاء الاصطناعي يتعلم من البيانات'},
        {'id': 3, 'text': 'الشبكة العصبية تحاكي الدماغ البشري'},
    ]
    
    question = 'ازاي النموذج بيفهم الأسئلة؟'
    
    # ردود مختلفة
    responses = [
        # رد منسوخ من السؤال
        'النموذج بيفهم الأسئلة',
        
        # رد منسوخ من Corpus
        'النموذج يفهم الأسئلة ويرد عليها',
        
        # رد مبتكر
        'النموذج بيحلل الكلمات ويستخرج المفاهيم وبيستخدم نظام الانتباه عشان يفهم السؤال كويس ويولد رد جديد',
        
        # رد متوسط
        'النموذج بيستخدم تقنيات متعددة للفهم',
    ]
    
    calculator = NoveltyCalculator(corpus=corpus)
    
    print("قياس أصالة الردود:\n")
    print("=" * 80)
    
    for i, response in enumerate(responses, 1):
        result = calculator.calculate(response, question)
        
        print(f"\nرد {i}:")
        print(f"  النص: {response}")
        print(f"  درجة الأصالة: {result.novelty_score:.3f}")
        print(f"  هل مبتكر؟: {'نعم ✓' if result.is_novel else 'لا ✗'}")
        
        if result.copied_phrases:
            print(f"  عبارات منسوخة: {', '.join(result.copied_phrases[:2])}")
        
        if result.matched_corpus_ids:
            print(f"  نصوص مطابقة: {result.matched_corpus_ids}")
        
        print(f"  تفاصيل التشابه:")
        for key, sim in result.similarity_breakdown.items():
            print(f"    {key}: {sim:.3f}")
    
    print("\n" + "=" * 80)
    print("\nتعزيز رد منخفض الأصالة:\n")
    
    low_novelty_response = 'النموذج يفهم الأسئلة ويرد عليها'
    result = calculator.calculate(low_novelty_response, question)
    
    print(f"الرد الأصلي: {low_novelty_response}")
    print(f"الأصالة: {result.novelty_score:.3f}")
    
    boosted = NoveltyBooster.boost_response(low_novelty_response, result)
    print(f"\nالرد المعزز: {boosted}")
