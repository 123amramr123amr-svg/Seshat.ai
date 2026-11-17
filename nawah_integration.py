"""
Nawah Integration for Seshat
دمج نواة مع seshat: استخدام البنية التحتية المتقدمة

Features:
- التحميل التلقائي لقاعدة معرفة seshat الحقيقية (data/ar_knowledge.txt وغيرها)
- Token Analysis للأسئلة العربية
- Attention System 16D لفهم السياق والربط المفاهيمي
- Concept Extraction للمعرفة (ثلاثيات فاعل-فعل-مفعول)
- Novelty Scoring للردود بدون استبدال إجابات seshat الدقيقة بقوالب عامة
- Database Logging للسجل في SQLite
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NAWAH_DIR = os.path.join(ROOT_DIR, 'nawah')
if NAWAH_DIR not in sys.path:
    sys.path.insert(0, NAWAH_DIR)

from pipeline import NawahPipeline, run_inference, InferenceResult
from database import get_db, KnowledgeCorpus


def load_seshat_corpus(max_docs: int = 250) -> list:
    """تحميل قاعدة المعرفة العربية الفعلية من ملفات data/ لتغذية نواة-1."""
    corpus = []
    data_dir = os.path.join(ROOT_DIR, 'data')
    sources = [
        ('ar_knowledge.txt', 'معرفة عامة'),
        ('ar_programming.txt', 'برمجة'),
        ('ar_chat.txt', 'محادثة'),
    ]
    doc_id = 1
    for filename, category in sources:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding='utf-8') as f:
                raw = f.read()
            blocks = [b.strip() for b in raw.split('\n\n') if b.strip()]
            for block in blocks:
                lines = [ln.strip() for ln in block.splitlines() if ln.strip() and not ln.strip().startswith('#')]
                if not lines:
                    continue
                q_line = lines[0]
                a_line = ' '.join(lines[1:]) if len(lines) > 1 else lines[0]
                q_clean = q_line.replace('س:', '').replace('Q:', '').strip()
                a_clean = a_line.replace('ج:', '').replace('A:', '').strip()
                if len(a_clean) < 10:
                    continue
                corpus.append({
                    'id': doc_id,
                    'title': q_clean[:80] or f'{category} #{doc_id}',
                    'content': f'{q_clean} — {a_clean}',
                    'category': category,
                })
                doc_id += 1
                if len(corpus) >= max_docs:
                    return corpus
        except Exception:
            continue
    return corpus


class NawahEnhancedBrain:
    """
    دماغ seshat المحسّنة بـ Nawah-1

    تضيف:
    - فهم عميق للأسئلة (Token Analysis + 16D Self-Attention)
    - استخراج المفاهيم الثلاثية (Concept Triplets)
    - قياس أصالة الردود (Novelty Score)
    - حفظ السجل في قاعدة البيانات SQLite
    """

    def __init__(self, corpus=None):
        self.corpus = corpus if corpus is not None else load_seshat_corpus()
        self.pipeline = NawahPipeline(corpus=self.corpus)
        self.db = get_db()

    def analyze_only(self, text: str, temperature: float = 0.75, save_to_db: bool = True) -> dict:
        """تشغيل خط تحليل نواة-1 الكامل وإرجاع النتائج المهيكلة."""
        result = self.pipeline.run(
            question=text,
            temperature=temperature,
            top_k=5,
            novelty_boost=0.85,
            save_to_db=save_to_db
        )
        return result.to_dict()

    def enhance_reply(
        self,
        question: str,
        base_reply: str,
        temperature: float = 0.75,
        is_fallback: bool = False
    ) -> dict:
        """
        تحسين رد seshat باستخدام Nawah-1 مع الحفاظ على ردود seshat الصحيحة.
        لا نستبدل رد seshat إلا إذا كان فارغًا أو في حالة fallback صريحة.
        """
        result = self.pipeline.run(
            question=question,
            temperature=temperature,
            top_k=5,
            novelty_boost=0.85,
            save_to_db=True
        )

        novelty_result = self.pipeline.novelty_calculator.calculate(
            response=base_reply or result.generated_response,
            question=question
        )

        # نحافظ على رد seshat الأساسي طالما ليس فارغًا وليس fallback
        if not base_reply or is_fallback:
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
                'top_concepts': [node for node, score in result.top_relevant_nodes[:5]],
                'extracted_concepts': [c.to_dict() for c in result.extracted_concepts],
                'analyzed_tokens': [
                    {
                        'token': t.token,
                        'normalized': t.normalized,
                        'stem': t.stem,
                        'pos': t.pos_label_ar,
                        'weight': t.semantic_weight
                    }
                    for t in result.analyzed_tokens
                ],
                'attention_matrix': result.attention_matrix.to_dict(),
                'latency_ms': result.latency_ms
            }
        }

    def add_knowledge(self, title: str, content: str, category: str = 'عام') -> int:
        """إضافة معرفة جديدة لقاعدة البيانات مع متجه دلالي 16D حقيقي."""
        concepts = self.pipeline.concept_extractor.extract_from_text(content)
        words = [w for w in content.split() if w.strip()]
        if words:
            vecs = self.pipeline.attention_system.computer.embedding.embed_batch(words[:32])
            dim = len(vecs[0])
            embedding = [round(sum(v[d] for v in vecs) / len(vecs), 5) for d in range(dim)]
        else:
            embedding = [0.0] * 16

        corpus_doc = KnowledgeCorpus(
            title=title,
            category=category,
            content=content,
            tokens_count=len(words),
            concepts_json=str([c.to_dict() for c in concepts]),
            embedding_vector=str(embedding),
            is_active=True
        )

        doc_id = self.db.add_corpus(corpus_doc)
        self.corpus.append({
            'id': doc_id,
            'title': title,
            'content': content,
            'category': category
        })
        self.pipeline = NawahPipeline(corpus=self.corpus)
        return doc_id

    def get_stats(self) -> dict:
        stats = dict(self.db.get_stats())
        stats['loaded_seshat_corpus_docs'] = len(self.corpus)
        return stats

    def get_recent_logs(self, limit: int = 10) -> list:
        return self.db.get_inference_logs(limit=limit)


_nawah_brain = None


def get_nawah_brain(corpus=None) -> NawahEnhancedBrain:
    """جلب نسخة NawahEnhancedBrain (Singleton)"""
    global _nawah_brain
    if _nawah_brain is None:
        _nawah_brain = NawahEnhancedBrain(corpus=corpus)
    return _nawah_brain
