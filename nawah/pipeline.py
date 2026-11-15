"""
Nawah Inference Pipeline
خط الاستنتاج الرئيسي: يجمع كل المكونات مع بعض

Pipeline Steps:
1. Token Analysis (تحليل الكلمات)
2. Concept Extraction (استخراج المفاهيم)
3. Attention Computation (حساب الانتباه)
4. Response Generation (توليد الرد)
5. Novelty Scoring (قياس الأصالة)
6. Database Logging (حفظ في قاعدة البيانات)
"""

import time
import json
import sys
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add nawah to path for imports
sys.path.insert(0, os.path.dirname(__file__))

try:
    from .token_analyzer import TokenAnalyzer, AnalyzedToken
    from .attention import AttentionSystem, AttentionMatrix
    from .concepts import TripletExtractor, ConceptTriplet, build_knowledge_graph
    from .novelty import NoveltyCalculator, NoveltyResult
    from .database import get_db, InferenceLog, KnowledgeCorpus
except ImportError:
    from token_analyzer import TokenAnalyzer, AnalyzedToken
    from attention import AttentionSystem, AttentionMatrix
    from concepts import TripletExtractor, ConceptTriplet, build_knowledge_graph
    from novelty import NoveltyCalculator, NoveltyResult
    from database import get_db, InferenceLog, KnowledgeCorpus


@dataclass
class InferenceResult:
    """نتيجة الاستنتاج الكاملة"""
    question: str
    generated_response: str
    
    # Analysis
    analyzed_tokens: List[AnalyzedToken]
    detected_intent: str
    intent_label_ar: str
    intent_confidence: float
    
    # Concepts
    extracted_concepts: List[ConceptTriplet]
    
    # Attention
    attention_matrix: AttentionMatrix
    top_relevant_nodes: List[tuple]
    
    # Novelty
    novelty_score: float
    is_novel: bool
    
    # Metadata
    latency_ms: int
    timestamp: str
    
    def to_dict(self) -> Dict:
        return {
            'question': self.question,
            'generated_response': self.generated_response,
            'analyzed_tokens': [
                {
                    'token': t.token,
                    'normalized': t.normalized,
                    'stem': t.stem,
                    'pos': t.pos,
                    'pos_label_ar': t.pos_label_ar,
                    'semantic_weight': t.semantic_weight
                }
                for t in self.analyzed_tokens
            ],
            'detected_intent': self.detected_intent,
            'intent_label_ar': self.intent_label_ar,
            'intent_confidence': self.intent_confidence,
            'extracted_concepts': [c.to_dict() for c in self.extracted_concepts],
            'attention_matrix': self.attention_matrix.to_dict(),
            'top_relevant_nodes': self.top_relevant_nodes,
            'novelty_score': self.novelty_score,
            'is_novel': self.is_novel,
            'latency_ms': self.latency_ms,
            'timestamp': self.timestamp
        }


class IntentDetector:
    """كاشف النية من السؤال"""
    
    INTENT_PATTERNS = {
        'DEFINITION': {
            'keywords': ['ما هو', 'ما هي', 'ماذا', 'تعريف', 'يعني ايه', 'هو ايه'],
            'label_ar': 'تعريف',
            'confidence': 0.9
        },
        'HOW_TO': {
            'keywords': ['ازاي', 'كيف', 'طريقة', 'خطوات'],
            'label_ar': 'كيفية',
            'confidence': 0.85
        },
        'WHY': {
            'keywords': ['ليه', 'لماذا', 'سبب'],
            'label_ar': 'سبب',
            'confidence': 0.85
        },
        'COMPARISON': {
            'keywords': ['فرق', 'مقارنة', 'أفضل من', 'أسوأ من'],
            'label_ar': 'مقارنة',
            'confidence': 0.8
        },
        'EXAMPLE': {
            'keywords': ['مثال', 'مثل', 'نموذج'],
            'label_ar': 'مثال',
            'confidence': 0.8
        },
        'GENERAL': {
            'keywords': [],
            'label_ar': 'عام',
            'confidence': 0.5
        }
    }
    
    @classmethod
    def detect(cls, question: str) -> tuple:
        """
        كشف النية من السؤال
        
        Returns:
            (intent, label_ar, confidence)
        """
        question_lower = question.lower()
        
        for intent, config in cls.INTENT_PATTERNS.items():
            for keyword in config['keywords']:
                if keyword in question_lower:
                    return intent, config['label_ar'], config['confidence']
        
        return 'GENERAL', 'عام', 0.5


class ResponseGenerator:
    """مولد الردود (مبسط)"""
    
    def __init__(self, knowledge_graph=None, corpus=None):
        self.knowledge_graph = knowledge_graph
        self.corpus = corpus or []
    
    def generate(
        self,
        question: str,
        intent: str,
        relevant_concepts: List[str],
        temperature: float = 0.75
    ) -> str:
        """
        توليد رد بناءً على السؤال والمفاهيم
        
        (هذا مولد مبسط - يمكن تطويره لاحقًا)
        """
        # استخراج الموضوع الرئيسي من السؤال
        topic = self._extract_main_topic(question, relevant_concepts)
        
        # البحث في Corpus عن معلومات
        relevant_info = self._search_corpus(question, topic)
        
        if relevant_info:
            return relevant_info
        
        # قوالب الردود حسب النية
        templates = {
            'DEFINITION': f'{topic} هو مفهوم مهم في مجال الذكاء الاصطناعي والتعلم الآلي',
            'HOW_TO': f'للتعامل مع {topic}، تحتاج إلى فهم الأساسيات والتدريب المستمر على البيانات',
            'WHY': f'{topic} مهم لأنه يساعد في فهم وتحليل البيانات وتوليد ردود ذكية',
            'COMPARISON': f'يمكن مقارنة {topic} مع المفاهيم الأخرى من حيث الأداء والدقة',
            'EXAMPLE': f'مثال على {topic}: النماذج اللغوية والشبكات العصبية',
            'GENERAL': f'بناءً على سؤالك عن {topic}، يمكن القول أنه موضوع مهم في مجال الذكاء الاصطناعي'
        }
        
        # اختيار القالب
        response = templates.get(intent, templates['GENERAL'])
        
        return response
    
    def _extract_main_topic(self, question: str, relevant_concepts: List[str]) -> str:
        """استخراج الموضوع الرئيسي من السؤال"""
        # محاولة استخراج أسماء من السؤال
        words = question.split()
        
        # البحث عن كلمات تبدأ بـ "ال"
        for word in words:
            if word.startswith('ال') and len(word) > 3:
                return word
        
        # استخدام أول مفهوم ذو صلة
        if relevant_concepts:
            # اختيار أطول كلمة (غالبًا الأكثر دلالة)
            return max(relevant_concepts[:3], key=len)
        
        return 'الموضوع'
    
    def _search_corpus(self, question: str, topic: str) -> Optional[str]:
        """البحث في Corpus عن معلومات ذات صلة"""
        if not self.corpus:
            return None
        
        # البحث عن وثائق تحتوي على الموضوع
        for doc in self.corpus:
            content = doc.get('content', '')
            if topic in content:
                # إرجاع جزء من المحتوى
                sentences = content.split('.')
                for sentence in sentences:
                    if topic in sentence and len(sentence) > 20:
                        return sentence.strip()
        
        return None


class NawahPipeline:
    """خط الاستنتاج الرئيسي"""
    
    def __init__(self, corpus: Optional[List[Dict]] = None):
        """
        Args:
            corpus: قائمة وثائق المعرفة
        """
        self.corpus = corpus or []
        
        # تهيئة المكونات
        self.token_analyzer = TokenAnalyzer(corpus=[c.get('content', '') for c in self.corpus])
        self.attention_system = AttentionSystem(embedding_dim=16, num_heads=2)
        self.concept_extractor = TripletExtractor()
        self.novelty_calculator = NoveltyCalculator(corpus=self.corpus)
        self.response_generator = ResponseGenerator()
        
        # بناء الرسم البياني المعرفي
        self.knowledge_graph = None
        if self.corpus:
            full_text = ' '.join([c.get('content', '') for c in self.corpus])
            self.knowledge_graph = build_knowledge_graph(full_text)
    
    def run(
        self,
        question: str,
        temperature: float = 0.75,
        top_k: int = 5,
        novelty_boost: float = 0.85,
        save_to_db: bool = True
    ) -> InferenceResult:
        """
        تشغيل خط الاستنتاج الكامل
        
        Args:
            question: السؤال
            temperature: درجة الحرارة (0-1)
            top_k: عدد المفاهيم المطلوبة
            novelty_boost: معامل تعزيز الأصالة
            save_to_db: هل نحفظ في قاعدة البيانات؟
        
        Returns:
            InferenceResult
        """
        start_time = time.time()
        
        # 1. تحليل الكلمات
        analyzed_tokens = self.token_analyzer.analyze(question)
        
        # 2. كشف النية
        intent, intent_label, intent_confidence = IntentDetector.detect(question)
        
        # 3. استخراج المفاهيم من السؤال
        extracted_concepts = self.concept_extractor.extract_from_text(question)
        
        # 4. بناء قائمة العقد المفاهيمية
        knowledge_nodes = self._build_knowledge_nodes()
        
        # 5. حساب الانتباه
        question_tokens = [t.token for t in analyzed_tokens]
        attention_matrix = self.attention_system.computer.compute_attention(
            query_tokens=question_tokens,
            key_tokens=knowledge_nodes
        )
        
        # 6. استخراج أهم العقد
        relevant_nodes = self.attention_system.get_relevant_nodes(
            attention_matrix,
            top_k=top_k
        )
        top_relevant = [node for node, score in relevant_nodes]
        
        # 7. توليد الرد
        response = self.response_generator.generate(
            question=question,
            intent=intent,
            relevant_concepts=top_relevant,
            temperature=temperature
        )
        
        # 8. قياس الأصالة
        novelty_result = self.novelty_calculator.calculate(
            response=response,
            question=question
        )
        
        # 9. حساب الوقت
        latency_ms = int((time.time() - start_time) * 1000)
        
        # 10. إنشاء النتيجة
        result = InferenceResult(
            question=question,
            generated_response=response,
            analyzed_tokens=analyzed_tokens,
            detected_intent=intent,
            intent_label_ar=intent_label,
            intent_confidence=intent_confidence,
            extracted_concepts=extracted_concepts,
            attention_matrix=attention_matrix,
            top_relevant_nodes=relevant_nodes,
            novelty_score=novelty_result.novelty_score,
            is_novel=novelty_result.is_novel,
            latency_ms=latency_ms,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 11. حفظ في قاعدة البيانات
        if save_to_db:
            self._save_to_db(result, temperature, top_k, novelty_boost)
        
        return result
    
    def _build_knowledge_nodes(self) -> List[str]:
        """بناء قائمة العقد المفاهيمية من قاعدة المعرفة"""
        nodes = []
        
        # إضافة كلمات من Corpus
        for doc in self.corpus:
            content = doc.get('content', '')
            words = content.split()
            nodes.extend(words[:20])  # أول 20 كلمة من كل وثيقة
        
        # إضافة مصطلحات أساسية
        base_terms = [
            'نموذج', 'ذكاء', 'اصطناعي', 'تعلم', 'آلي',
            'شبكة', 'عصبية', 'بيانات', 'تدريب', 'انتباه',
            'متجه', 'دلالة', 'كلمة', 'جملة', 'نص',
            'فهم', 'سؤال', 'رد', 'تحليل', 'مفهوم'
        ]
        nodes.extend(base_terms)
        
        # إزالة التكرار
        return list(set(nodes))
    
    def _save_to_db(
        self,
        result: InferenceResult,
        temperature: float,
        top_k: int,
        novelty_boost: float
    ):
        """حفظ النتيجة في قاعدة البيانات"""
        try:
            db = get_db()
            
            log = InferenceLog(
                question=result.question,
                detected_intent=result.detected_intent,
                intent_label_ar=result.intent_label_ar,
                intent_confidence=result.intent_confidence,
                extracted_entities_json=json.dumps([
                    {
                        'token': t.token,
                        'normalized': t.normalized,
                        'stem': t.stem,
                        'pos': t.pos,
                        'semantic_weight': t.semantic_weight
                    }
                    for t in result.analyzed_tokens
                ]),
                attention_matrix_json=json.dumps(result.attention_matrix.to_dict()),
                generated_response=result.generated_response,
                novelty_score=result.novelty_score,
                synthesized_clauses_json=json.dumps([
                    c.to_dict() for c in result.extracted_concepts
                ]),
                temperature_used=temperature,
                top_k_used=top_k,
                novelty_boost_used=novelty_boost,
                latency_ms=result.latency_ms
            )
            
            db.add_inference_log(log)
        except Exception as e:
            print(f"⚠️ فشل حفظ في قاعدة البيانات: {e}")


# Convenience function
def run_inference(
    question: str,
    corpus: Optional[List[Dict]] = None,
    temperature: float = 0.75,
    top_k: int = 5,
    novelty_boost: float = 0.85
) -> InferenceResult:
    """
    دالة مساعدة لتشغيل الاستنتاج
    
    Args:
        question: السؤال
        corpus: قائمة وثائق المعرفة
        temperature: درجة الحرارة
        top_k: عدد المفاهيم
        novelty_boost: معامل الأصالة
    
    Returns:
        InferenceResult
    """
    pipeline = NawahPipeline(corpus=corpus)
    return pipeline.run(
        question=question,
        temperature=temperature,
        top_k=top_k,
        novelty_boost=novelty_boost
    )


# Test
if __name__ == '__main__':
    print("اختبار Nawah Pipeline:\n")
    print("=" * 80)
    
    # قاعدة معرفة تجريبية
    test_corpus = [
        {
            'id': 1,
            'title': 'مقدمة في الذكاء الاصطناعي',
            'content': 'الذكاء الاصطناعي هو فرع من علوم الحاسب يهدف إلى إنشاء أنظمة قادرة على محاكاة الذكاء البشري والتعلم من البيانات'
        },
        {
            'id': 2,
            'title': 'الشبكات العصبية',
            'content': 'الشبكات العصبية هي نماذج حسابية مستوحاة من الدماغ البشري تتكون من عقد مترابطة تعالج المعلومات'
        },
        {
            'id': 3,
            'title': 'التعلم الآلي',
            'content': 'التعلم الآلي هو subset من الذكاء الاصطناعي يركز على تطوير خوارزميات تتعلم من البيانات وتحسن أداءها تلقائياً'
        }
    ]
    
    # أسئلة تجريبية
    test_questions = [
        'ما هو الذكاء الاصطناعي؟',
        'ازاي الشبكة العصبية بتشتغل؟',
        'ليه التعلم الآلي مهم؟',
        'يعني ايه نموذج ذكاء اصطناعي؟'
    ]
    
    for question in test_questions:
        print(f"\n❓ السؤال: {question}\n")
        
        result = run_inference(
            question=question,
            corpus=test_corpus,
            temperature=0.75,
            top_k=5
        )
        
        print(f"💬 الرد: {result.generated_response}\n")
        
        print(f"📊 التحليل:")
        print(f"   النية: {result.intent_label_ar} (ثقة: {result.intent_confidence:.2f})")
        print(f"   الأصالة: {result.novelty_score:.3f} ({'مبتكر ✓' if result.is_novel else 'منسوخ ✗'})")
        print(f"   الوقت: {result.latency_ms}ms")
        
        print(f"\n🔍 أهم المفاهيم:")
        for node, score in result.top_relevant_nodes[:3]:
            print(f"   {node}: {score:.4f}")
        
        print(f"\n📝 الكلمات المحللة ({len(result.analyzed_tokens)}):")
        for token in result.analyzed_tokens[:3]:
            print(f"   {token.token} → {token.stem} ({token.pos_label_ar})")
        
        print("\n" + "-" * 80)
    
    print("\n✅ تم اختبار Pipeline بنجاح!")
