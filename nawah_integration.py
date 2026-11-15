"""
Nawah Integration for Seshat
دمج نواة مع seshat: استخدام البنية التحتية المتقدمة

Features:
- Token Analysis للأسئلة
- Attention System لفهم السياق
- Concept Extraction للمعرفة
- Novelty Scoring للردود
- Database Logging للسجل
"""

import sys
import os

# Add nawah to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nawah'))

from pipeline import NawahPipeline, run_inference, InferenceResult
from database import get_db, KnowledgeCorpus


class NawahEnhancedBrain:
    """
    دماغ seshat المحسّنة بـ Nawah
    
    تضيف:
    - فهم عميق للأسئلة (Token Analysis + Attention)
    - قياس أصالة الردود (Novelty Score)
    - حفظ السجل في قاعدة البيانات
    """
    
    def __init__(self, corpus=None):
        """
        Args:
            corpus: قائمة وثائق المعرفة من seshat
        """
        self.corpus = corpus or []
        self.pipeline = NawahPipeline(corpus=self.corpus)
        self.db = get_db()
    
    def enhance_reply(
        self,
        question: str,
        base_reply: str,
        temperature: float = 0.75
    ) -> dict:
        """
        تحسين رد seshat باستخدام Nawah
        
        Args:
            question: السؤال الأصلي
            base_reply: الرد الأساسي من seshat
            temperature: درجة الحرارة
        
        Returns:
            dict مع الرد المحسن والتحليل
        """
        # تشغيل Nawah Pipeline
        result = self.pipeline.run(
            question=question,
            temperature=temperature,
            top_k=5,
            novelty_boost=0.85,
            save_to_db=True
        )
        
        # قياس أصالة الرد الأساسي
        novelty_result = self.pipeline.novelty_calculator.calculate(
            response=base_reply,
            question=question
        )
        
        # إذا الرد الأساسي منخفض الأصالة، استخدم رد Nawah
        if novelty_result.novelty_score < 0.5:
            enhanced_reply = result.generated_response
        else:
            enhanced_reply = base_reply
        
        return {
            'reply': enhanced_reply,
            'analysis': {
                'intent': result.detected_intent,
                'intent_label': result.intent_label_ar,
                'intent_confidence': result.intent_confidence,
                'novelty_score': novelty_result.novelty_score,
                'is_novel': novelty_result.is_novel,
                'top_concepts': [node for node, score in result.top_relevant_nodes[:3]],
                'analyzed_tokens': [
                    {
                        'token': t.token,
                        'stem': t.stem,
                        'pos': t.pos_label_ar,
                        'weight': t.semantic_weight
                    }
                    for t in result.analyzed_tokens
                ],
                'latency_ms': result.latency_ms
            }
        }
    
    def add_knowledge(self, title: str, content: str, category: str = 'عام') -> int:
        """
        إضافة معرفة جديدة لقاعدة البيانات
        
        Args:
            title: عنوان الوثيقة
            content: محتوى الوثيقة
            category: التصنيف
        
        Returns:
            معرف الوثيقة
        """
        # استخراج المفاهيم
        concepts = self.pipeline.concept_extractor.extract_from_text(content)
        
        # حساب المتجه الدلالي (مبسط)
        words = content.split()
        embedding = [0.1 * i for i in range(16)]  # متجه تجريبي
        
        # إنشاء الوثيقة
        corpus_doc = KnowledgeCorpus(
            title=title,
            category=category,
            content=content,
            tokens_count=len(words),
            concepts_json=str([c.to_dict() for c in concepts]),
            embedding_vector=str(embedding),
            is_active=True
        )
        
        # حفظ في قاعدة البيانات
        doc_id = self.db.add_corpus(corpus_doc)
        
        # تحديث الـ corpus في الـ pipeline
        self.corpus.append({
            'id': doc_id,
            'title': title,
            'content': content,
            'category': category
        })
        
        # إعادة تهيئة الـ pipeline
        self.pipeline = NawahPipeline(corpus=self.corpus)
        
        return doc_id
    
    def get_stats(self) -> dict:
        """جلب إحصائيات النظام"""
        return self.db.get_stats()
    
    def get_recent_logs(self, limit: int = 10) -> list:
        """جلب آخر الاستنتاجات"""
        return self.db.get_inference_logs(limit=limit)


# Singleton instance
_nawah_brain = None

def get_nawah_brain(corpus=None) -> NawahEnhancedBrain:
    """جلب نسخة NawahEnhancedBrain (Singleton)"""
    global _nawah_brain
    if _nawah_brain is None:
        _nawah_brain = NawahEnhancedBrain(corpus=corpus)
    return _nawah_brain


# Test
if __name__ == '__main__':
    print("اختبار Nawah Integration with Seshat:\n")
    print("=" * 80)
    
    # قاعدة معرفة تجريبية
    test_corpus = [
        {
            'id': 1,
            'title': 'مقدمة في الذكاء الاصطناعي',
            'content': 'الذكاء الاصطناعي هو فرع من علوم الحاسب يهدف إلى إنشاء أنظمة قادرة على محاكاة الذكاء البشري'
        }
    ]
    
    brain = NawahEnhancedBrain(corpus=test_corpus)
    
    # اختبار تحسين رد
    question = 'ما هو الذكاء الاصطناعي؟'
    base_reply = 'الذكاء الاصطناعي هو مجال في علوم الحاسب'
    
    print(f"❓ السؤال: {question}")
    print(f"💬 الرد الأساسي: {base_reply}\n")
    
    enhanced = brain.enhance_reply(question, base_reply)
    
    print(f"✨ الرد المحسن: {enhanced['reply']}\n")
    
    print(f"📊 التحليل:")
    analysis = enhanced['analysis']
    print(f"   النية: {analysis['intent_label']} (ثقة: {analysis['intent_confidence']:.2f})")
    print(f"   الأصالة: {analysis['novelty_score']:.3f}")
    print(f"   أهم المفاهيم: {', '.join(analysis['top_concepts'])}")
    print(f"   الوقت: {analysis['latency_ms']}ms")
    
    print(f"\n📝 الكلمات المحللة:")
    for token in analysis['analyzed_tokens'][:5]:
        print(f"   {token['token']} → {token['stem']} ({token['pos']}, وزن: {token['weight']:.2f})")
    
    # اختبار إضافة معرفة
    print("\n" + "=" * 80)
    print("\n➕ إضافة معرفة جديدة:\n")
    
    doc_id = brain.add_knowledge(
        title='الشبكات العصبية',
        content='الشبكات العصبية هي نماذج حسابية مستوحاة من الدماغ البشري',
        category='تعلم آلي'
    )
    
    print(f"✓ تم إضافة وثيقة بمعرف: {doc_id}")
    
    # إحصائيات
    print("\n📈 الإحصائيات:")
    stats = brain.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ تم اختبار Integration بنجاح!")
