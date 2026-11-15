"""
Nawah Database - SQLite Persistence
قاعدة بيانات SQLite لحفظ كل حاجة

Tables:
- inference_logs: سجل الاستنتاجات (الأسئلة والردود)
- knowledge_corpus: قاعدة المعرفة (النصوص التدريبية)
- vocabulary: المعجم (الكلمات وجذورها)
- model_checkpoints: نقاط حفظ النموذج
"""

import sqlite3
import json
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), 'nawah.db')


@dataclass
class InferenceLog:
    """سجل استنتاج واحد"""
    id: Optional[int] = None
    question: str = ''
    detected_intent: str = ''
    intent_label_ar: str = ''
    intent_confidence: float = 0.0
    extracted_entities_json: str = '[]'  # JSON
    attention_matrix_json: str = '{}'    # JSON
    generated_response: str = ''
    novelty_score: float = 0.0
    synthesized_clauses_json: str = '[]' # JSON
    temperature_used: float = 0.75
    top_k_used: int = 5
    novelty_boost_used: float = 0.85
    latency_ms: int = 0
    user_rating: Optional[int] = None
    created_at: str = ''
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class KnowledgeCorpus:
    """وثيقة في قاعدة المعرفة"""
    id: Optional[int] = None
    title: str = ''
    category: str = ''
    content: str = ''
    tokens_count: int = 0
    concepts_json: str = '[]'        # JSON (ConceptTriplet[])
    embedding_vector: str = '[]'     # JSON (float[])
    is_active: bool = True
    created_at: str = ''
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class VocabularyEntry:
    """كلمة في المعجم"""
    id: Optional[int] = None
    token: str = ''
    root_stem: str = ''
    pos_tag: str = ''
    frequency: int = 0
    idf_weight: float = 0.0
    semantic_coords: str = '[]'      # JSON (float[])
    transitions_json: str = '{}'     # JSON (token -> probability)
    created_at: str = ''
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelCheckpoint:
    """نقطة حفظ النموذج"""
    id: Optional[int] = None
    version_name: str = ''
    vocab_size: int = 0
    corpus_size: int = 0
    embedding_dim: int = 16
    attention_heads: int = 2
    context_window: int = 64
    perplexity: float = 0.0
    loss: float = 0.0
    training_epochs: int = 0
    transition_edges_count: int = 0
    hyperparams_json: str = '{}'     # JSON
    loss_history_json: str = '[]'    # JSON
    created_at: str = ''
    
    def to_dict(self) -> Dict:
        return asdict(self)


class NawahDatabase:
    """قاعدة بيانات نواة"""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """الاتصال بقاعدة البيانات"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys = ON')
    
    def _create_tables(self):
        """إنشاء الجداول"""
        cursor = self.conn.cursor()
        
        # جدول سجل الاستنتاجات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                detected_intent TEXT,
                intent_label_ar TEXT,
                intent_confidence REAL,
                extracted_entities_json TEXT,
                attention_matrix_json TEXT,
                generated_response TEXT,
                novelty_score REAL,
                synthesized_clauses_json TEXT,
                temperature_used REAL,
                top_k_used INTEGER,
                novelty_boost_used REAL,
                latency_ms INTEGER,
                user_rating INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول قاعدة المعرفة
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_corpus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT,
                content TEXT NOT NULL,
                tokens_count INTEGER,
                concepts_json TEXT,
                embedding_vector TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول المعجم
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                token TEXT NOT NULL UNIQUE,
                root_stem TEXT,
                pos_tag TEXT,
                frequency INTEGER DEFAULT 1,
                idf_weight REAL,
                semantic_coords TEXT,
                transitions_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول نقاط الحفظ
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_name TEXT NOT NULL,
                vocab_size INTEGER,
                corpus_size INTEGER,
                embedding_dim INTEGER DEFAULT 16,
                attention_heads INTEGER DEFAULT 2,
                context_window INTEGER DEFAULT 64,
                perplexity REAL,
                loss REAL,
                training_epochs INTEGER,
                transition_edges_count INTEGER,
                hyperparams_json TEXT,
                loss_history_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    # ==================== Inference Logs ====================
    
    def add_inference_log(self, log: InferenceLog) -> int:
        """إضافة سجل استنتاج"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO inference_logs (
                question, detected_intent, intent_label_ar, intent_confidence,
                extracted_entities_json, attention_matrix_json, generated_response,
                novelty_score, synthesized_clauses_json, temperature_used,
                top_k_used, novelty_boost_used, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log.question, log.detected_intent, log.intent_label_ar, log.intent_confidence,
            log.extracted_entities_json, log.attention_matrix_json, log.generated_response,
            log.novelty_score, log.synthesized_clauses_json, log.temperature_used,
            log.top_k_used, log.novelty_boost_used, log.latency_ms
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_inference_logs(self, limit: int = 50) -> List[Dict]:
        """جلب سجلات الاستنتاج"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM inference_logs ORDER BY id DESC LIMIT ?', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def update_inference_rating(self, log_id: int, rating: int) -> bool:
        """تحديث تقييم استنتاج"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE inference_logs SET user_rating = ? WHERE id = ?', (rating, log_id))
        self.conn.commit()
        return cursor.rowcount > 0
    
    # ==================== Knowledge Corpus ====================
    
    def add_corpus(self, corpus: KnowledgeCorpus) -> int:
        """إضافة وثيقة لقاعدة المعرفة"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO knowledge_corpus (
                title, category, content, tokens_count,
                concepts_json, embedding_vector, is_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            corpus.title, corpus.category, corpus.content, corpus.tokens_count,
            corpus.concepts_json, corpus.embedding_vector, corpus.is_active
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_corpus(self, active_only: bool = True) -> List[Dict]:
        """جلب وثائق قاعدة المعرفة"""
        cursor = self.conn.cursor()
        if active_only:
            cursor.execute('SELECT * FROM knowledge_corpus WHERE is_active = 1 ORDER BY id DESC')
        else:
            cursor.execute('SELECT * FROM knowledge_corpus ORDER BY id DESC')
        return [dict(row) for row in cursor.fetchall()]
    
    def toggle_corpus(self, corpus_id: int, is_active: bool) -> bool:
        """تفعيل/تعطيل وثيقة"""
        cursor = self.conn.cursor()
        cursor.execute('UPDATE knowledge_corpus SET is_active = ? WHERE id = ?', (is_active, corpus_id))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def delete_corpus(self, corpus_id: int) -> bool:
        """حذف وثيقة"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM knowledge_corpus WHERE id = ?', (corpus_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    # ==================== Vocabulary ====================
    
    def add_or_update_vocabulary(self, entry: VocabularyEntry) -> int:
        """إضافة أو تحديث كلمة في المعجم"""
        cursor = self.conn.cursor()
        
        # محاولة التحديث أولاً
        cursor.execute('''
            UPDATE vocabulary SET
                frequency = frequency + 1,
                root_stem = ?,
                pos_tag = ?,
                idf_weight = ?,
                semantic_coords = ?,
                transitions_json = ?
            WHERE token = ?
        ''', (
            entry.root_stem, entry.pos_tag, entry.idf_weight,
            entry.semantic_coords, entry.transitions_json, entry.token
        ))
        
        if cursor.rowcount == 0:
            # إدراج جديد
            cursor.execute('''
                INSERT INTO vocabulary (
                    token, root_stem, pos_tag, frequency, idf_weight,
                    semantic_coords, transitions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.token, entry.root_stem, entry.pos_tag, entry.frequency,
                entry.idf_weight, entry.semantic_coords, entry.transitions_json
            ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_vocabulary(self, limit: int = 100, order_by: str = 'frequency') -> List[Dict]:
        """جلب المعجم"""
        cursor = self.conn.cursor()
        cursor.execute(f'SELECT * FROM vocabulary ORDER BY {order_by} DESC LIMIT ?', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_vocabulary_count(self) -> int:
        """عدد الكلمات في المعجم"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM vocabulary')
        return cursor.fetchone()[0]
    
    # ==================== Model Checkpoints ====================
    
    def add_checkpoint(self, checkpoint: ModelCheckpoint) -> int:
        """إضافة نقطة حفظ"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO model_checkpoints (
                version_name, vocab_size, corpus_size, embedding_dim,
                attention_heads, context_window, perplexity, loss,
                training_epochs, transition_edges_count,
                hyperparams_json, loss_history_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            checkpoint.version_name, checkpoint.vocab_size, checkpoint.corpus_size,
            checkpoint.embedding_dim, checkpoint.attention_heads, checkpoint.context_window,
            checkpoint.perplexity, checkpoint.loss, checkpoint.training_epochs,
            checkpoint.transition_edges_count, checkpoint.hyperparams_json,
            checkpoint.loss_history_json
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_checkpoints(self, limit: int = 10) -> List[Dict]:
        """جلب نقاط الحفظ"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM model_checkpoints ORDER BY id DESC LIMIT ?', (limit,))
        return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_checkpoint(self) -> Optional[Dict]:
        """جلب آخر نقطة حفظ"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM model_checkpoints ORDER BY id DESC LIMIT 1')
        row = cursor.fetchone()
        return dict(row) if row else None
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict:
        """إحصائيات عامة"""
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute('SELECT COUNT(*) FROM inference_logs')
        stats['inference_logs_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM knowledge_corpus WHERE is_active = 1')
        stats['active_corpus_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM vocabulary')
        stats['vocabulary_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM model_checkpoints')
        stats['checkpoints_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(novelty_score) FROM inference_logs WHERE novelty_score > 0')
        avg_novelty = cursor.fetchone()[0]
        stats['avg_novelty_score'] = avg_novelty if avg_novelty else 0.0
        
        return stats
    
    def close(self):
        """إغلاق الاتصال"""
        if self.conn:
            self.conn.close()


# Singleton instance
_db_instance = None

def get_db() -> NawahDatabase:
    """جلب نسخة قاعدة البيانات (Singleton)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = NawahDatabase()
    return _db_instance


# Test
if __name__ == '__main__':
    # اختبار قاعدة البيانات
    print("اختبار قاعدة بيانات نواة:\n")
    
    # حذف قاعدة البيانات القديمة للاختبار
    if os.path.exists('nawah.db'):
        os.remove('nawah.db')
    
    db = NawahDatabase()
    
    # 1. إضافة وثيقة معرفة
    print("1. إضافة وثيقة معرفة...")
    corpus = KnowledgeCorpus(
        title='مقدمة في الذكاء الاصطناعي',
        category='علوم الحاسب',
        content='الذكاء الاصطناعي هو فرع من علوم الحاسب يهدف إلى إنشاء أنظمة قادرة على محاكاة الذكاء البشري',
        tokens_count=15,
        concepts_json=json.dumps([{'subject': 'ذكاء', 'predicate': 'يهدف', 'object': 'محاكاة'}]),
        embedding_vector=json.dumps([0.1, 0.2, 0.3]),
        is_active=True
    )
    corpus_id = db.add_corpus(corpus)
    print(f"   ✓ تم إضافة وثيقة بمعرف: {corpus_id}")
    
    # 2. إضافة كلمة للمعجم
    print("\n2. إضافة كلمة للمعجم...")
    vocab = VocabularyEntry(
        token='ذكاء',
        root_stem='ذكا',
        pos_tag='NOUN',
        frequency=5,
        idf_weight=2.3,
        semantic_coords=json.dumps([0.5, 0.6, 0.7])
    )
    vocab_id = db.add_or_update_vocabulary(vocab)
    print(f"   ✓ تم إضافة كلمة بمعرف: {vocab_id}")
    
    # 3. إضافة سجل استنتاج
    print("\n3. إضافة سجل استنتاج...")
    log = InferenceLog(
        question='ما هو الذكاء الاصطناعي؟',
        detected_intent='DEFINITION',
        intent_label_ar='تعريف',
        intent_confidence=0.92,
        extracted_entities_json=json.dumps([{'token': 'ذكاء', 'pos': 'NOUN'}]),
        attention_matrix_json=json.dumps({'weights': [[0.8, 0.2]]}),
        generated_response='الذكاء الاصطناعي هو فرع من علوم الحاسب...',
        novelty_score=0.85,
        temperature_used=0.75,
        top_k_used=5,
        latency_ms=45
    )
    log_id = db.add_inference_log(log)
    print(f"   ✓ تم إضافة سجل بمعرف: {log_id}")
    
    # 4. إضافة نقطة حفظ
    print("\n4. إضافة نقطة حفظ...")
    checkpoint = ModelCheckpoint(
        version_name='Nawah-1.001-Initial',
        vocab_size=100,
        corpus_size=5,
        embedding_dim=16,
        attention_heads=2,
        perplexity=45.2,
        loss=0.32,
        training_epochs=10,
        hyperparams_json=json.dumps({'learning_rate': 0.0025})
    )
    checkpoint_id = db.add_checkpoint(checkpoint)
    print(f"   ✓ تم إضافة نقطة حفظ بمعرف: {checkpoint_id}")
    
    # 5. جلب الإحصائيات
    print("\n5. الإحصائيات:")
    stats = db.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 6. جلب البيانات
    print("\n6. جلب البيانات:")
    
    logs = db.get_inference_logs(limit=5)
    print(f"   سجلات الاستنتاج: {len(logs)}")
    
    corpus_docs = db.get_corpus()
    print(f"   وثائق المعرفة: {len(corpus_docs)}")
    
    vocab_list = db.get_vocabulary(limit=10)
    print(f"   المعجم: {len(vocab_list)} كلمة")
    
    checkpoints = db.get_checkpoints(limit=5)
    print(f"   نقاط الحفظ: {len(checkpoints)}")
    
    db.close()
    print("\n✅ تم اختبار قاعدة البيانات بنجاح!")
