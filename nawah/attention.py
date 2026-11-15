"""
Self-Attention System for Nawah
نظام الانتباه الذاتي: حساب مصفوفة الانتباه Q×Kᵀ/√d

Features:
- Multi-head attention (simplified)
- Query-Key-Value computation
- Softmax normalization
- Attention visualization
- Weighted context aggregation
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AttentionMatrix:
    """مصفوفة الانتباه"""
    query_tokens: List[str]      # كلمات السؤال
    concept_nodes: List[str]     # العقد المفاهيمية
    weights: List[List[float]]   # مصفوفة الأوزان (Q × K)
    
    def to_dict(self) -> Dict:
        return {
            'query_tokens': self.query_tokens,
            'concept_nodes': self.concept_nodes,
            'weights': self.weights
        }


class Softmax:
    """دالة Softmax للتطبيع"""
    
    @staticmethod
    def compute(x: List[float]) -> List[float]:
        """
        Softmax: exp(x_i) / sum(exp(x_j))
        """
        max_x = max(x)  # للاستقرار العددي
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [e / sum_exp for e in exp_x]


class SemanticEmbedding:
    """تمثيل دلالي للكلمات (Embedding)"""
    
    def __init__(self, dim: int = 16):
        """
        Args:
            dim: أبعاد المتجه الدلالي (16D مثل نواة-1)
        """
        self.dim = dim
        self.cache: Dict[str, np.ndarray] = {}
    
    def embed(self, token: str) -> np.ndarray:
        """
        تحويل كلمة إلى متجه دلالي
        
        نستخدم hash-based embedding بسيط:
        - كل كلمة ليها متجه ثابت (deterministic)
        - ممكن نطورها لاحقًا بـ word2vec أو GloVe
        """
        if token in self.cache:
            return self.cache[token]
        
        # Hash-based embedding
        np.random.seed(hash(token) % (2**32))
        vec = np.random.randn(self.dim)
        
        # تطبيع (normalize)
        vec = vec / np.linalg.norm(vec)
        
        self.cache[token] = vec
        return vec
    
    def embed_batch(self, tokens: List[str]) -> np.ndarray:
        """تحويل قائمة كلمات إلى مصفوفة متجهات"""
        return np.array([self.embed(t) for t in tokens])


class AttentionComputer:
    """حساب الانتباه الذاتي"""
    
    def __init__(self, embedding_dim: int = 16, num_heads: int = 2):
        """
        Args:
            embedding_dim: أبعاد المتجه الدلالي
            num_heads: عدد رؤوس الانتباه (multi-head)
        """
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding = SemanticEmbedding(dim=embedding_dim)
        
        # مصفوفات التحويل (Q, K, V) - مبسطة
        np.random.seed(42)
        self.W_q = np.random.randn(embedding_dim, embedding_dim) / math.sqrt(embedding_dim)
        self.W_k = np.random.randn(embedding_dim, embedding_dim) / math.sqrt(embedding_dim)
        self.W_v = np.random.randn(embedding_dim, embedding_dim) / math.sqrt(embedding_dim)
    
    def compute_attention(
        self,
        query_tokens: List[str],
        key_tokens: List[str]
    ) -> AttentionMatrix:
        """
        حساب مصفوفة الانتباه بين كلمات السؤال والعقد المفاهيمية
        
        Attention(Q, K) = softmax(Q × Kᵀ / √d)
        
        Args:
            query_tokens: كلمات السؤال (Q)
            key_tokens: العقد المفاهيمية (K)
        
        Returns:
            AttentionMatrix مع الأوزان
        """
        # تحويل الكلمات إلى متجهات
        Q = self.embedding.embed_batch(query_tokens)  # (num_queries, dim)
        K = self.embedding.embed_batch(key_tokens)    # (num_keys, dim)
        
        # تطبيق مصفوفات التحويل
        Q = Q @ self.W_q  # (num_queries, dim)
        K = K @ self.W_k  # (num_keys, dim)
        
        # حساب الانتباه: Q × Kᵀ
        attention_scores = Q @ K.T  # (num_queries, num_keys)
        
        # التطبيع بـ √d
        scale_factor = math.sqrt(self.embedding_dim)
        attention_scores = attention_scores / scale_factor
        
        # Softmax لكل query
        attention_weights = []
        for i in range(len(query_tokens)):
            weights = Softmax.compute(attention_scores[i].tolist())
            attention_weights.append(weights)
        
        return AttentionMatrix(
            query_tokens=query_tokens,
            concept_nodes=key_tokens,
            weights=attention_weights
        )
    
    def compute_weighted_context(
        self,
        query_tokens: List[str],
        value_tokens: List[str],
        attention_matrix: AttentionMatrix
    ) -> np.ndarray:
        """
        حساب السياق الموزون بالأنتباه
        
        Context = Attention × V
        
        Args:
            query_tokens: كلمات السؤال
            value_tokens: العقد المفاهيمية (V)
            attention_matrix: مصفوفة الانتباه المحسوبة
        
        Returns:
            متجه السياق الموزون
        """
        # تحويل القيم إلى متجهات
        V = self.embedding.embed_batch(value_tokens)  # (num_values, dim)
        V = V @ self.W_v  # تطبيق مصفوفة التحويل
        
        # تحويل مصفوفة الانتباه إلى numpy
        attention_weights = np.array(attention_matrix.weights)  # (num_queries, num_values)
        
        # حساب السياق الموزون
        context = attention_weights @ V  # (num_queries, dim)
        
        # تجميع السياق لكل الاستعلامات
        aggregated_context = np.mean(context, axis=0)  # (dim,)
        
        return aggregated_context


class ConceptExtractor:
    """استخراج المفاهيم من النص"""
    
    @staticmethod
    def extract_concepts(tokens: List[str], pos_tags: List[str]) -> List[str]:
        """
        استخراج العقد المفاهيمية من النص
        
        نختار:
        - الأسماء (NOUN)
        - المصطلحات التقنية (TECH_TERM)
        - الأفعال المهمة (VERB)
        """
        concepts = []
        
        for token, pos in zip(tokens, pos_tags):
            if pos in ['NOUN', 'TECH_TERM', 'VERB']:
                concepts.append(token)
        
        # إزالة التكرار مع الحفاظ على الترتيب
        seen = set()
        unique_concepts = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique_concepts.append(c)
        
        return unique_concepts


class AttentionSystem:
    """نظام الانتباه الرئيسي"""
    
    def __init__(self, embedding_dim: int = 16, num_heads: int = 2):
        self.computer = AttentionComputer(embedding_dim, num_heads)
    
    def analyze_question(
        self,
        question_tokens: List[str],
        knowledge_nodes: List[str]
    ) -> Tuple[AttentionMatrix, np.ndarray]:
        """
        تحليل سؤال باستخدام نظام الانتباه
        
        Args:
            question_tokens: كلمات السؤال
            knowledge_nodes: العقد المفاهيمية من قاعدة المعرفة
        
        Returns:
            (AttentionMatrix, weighted_context)
        """
        # حساب مصفوفة الانتباه
        attention = self.computer.compute_attention(
            query_tokens=question_tokens,
            key_tokens=knowledge_nodes
        )
        
        # حساب السياق الموزون
        context = self.computer.compute_weighted_context(
            query_tokens=question_tokens,
            value_tokens=knowledge_nodes,
            attention_matrix=attention
        )
        
        return attention, context
    
    def get_relevant_nodes(
        self,
        attention_matrix: AttentionMatrix,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        استخراج أهم العقد المفاهيمية بناءً على الانتباه
        
        Args:
            attention_matrix: مصفوفة الانتباه
            top_k: عدد العقد المطلوبة
        
        Returns:
            قائمة (node, score) مرتبة حسب الأهمية
        """
        # حساب متوسط الانتباه لكل عقدة
        node_scores = {}
        
        for node_idx, node in enumerate(attention_matrix.concept_nodes):
            # متوسط الانتباه من كل كلمات السؤال
            avg_attention = sum(
                attention_matrix.weights[q_idx][node_idx]
                for q_idx in range(len(attention_matrix.query_tokens))
            ) / len(attention_matrix.query_tokens)
            
            node_scores[node] = avg_attention
        
        # ترتيب حسب الأهمية
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:top_k]


# Convenience function
def compute_attention(
    question: str,
    knowledge_base: List[str],
    embedding_dim: int = 16
) -> Tuple[AttentionMatrix, List[Tuple[str, float]]]:
    """
    دالة مساعدة لحساب الانتباه
    
    Args:
        question: السؤال
        knowledge_base: قائمة العقد المفاهيمية
        embedding_dim: أبعاد المتجه
    
    Returns:
        (AttentionMatrix, top_relevant_nodes)
    """
    system = AttentionSystem(embedding_dim=embedding_dim)
    
    # تقسيم السؤال إلى كلمات
    question_tokens = question.split()
    
    # حساب الانتباه
    attention, context = system.analyze_question(
        question_tokens=question_tokens,
        knowledge_nodes=knowledge_base
    )
    
    # استخراج أهم العقد
    relevant_nodes = system.get_relevant_nodes(attention, top_k=5)
    
    return attention, relevant_nodes


# Test
if __name__ == '__main__':
    # مثال على الاستخدام
    question = 'ازاي النموذج بيفهم الأسئلة؟'
    
    knowledge_base = [
        'نموذج', 'ذكاء', 'اصطناعي', 'فهم', 'سؤال', 'رد',
        'تعلم', 'آلي', 'شبكة', 'عصبية', 'بيانات', 'تدريب',
        'انتباه', 'متجه', 'دلالة', 'كلمة', 'جملة', 'نص'
    ]
    
    print(f"السؤال: {question}\n")
    print(f"قاعدة المعرفة: {len(knowledge_base)} عقدة\n")
    
    attention, relevant = compute_attention(question, knowledge_base)
    
    print("مصفوفة الانتباه:")
    print(f"  كلمات السؤال: {attention.query_tokens}")
    print(f"  العقد المفاهيمية: {attention.concept_nodes[:5]}...")
    print(f"  الأبعاد: {len(attention.weights)} × {len(attention.weights[0])}\n")
    
    print("أهم العقد المفاهيمية:")
    for node, score in relevant:
        print(f"  {node:<15} {score:.4f}")
    
    print("\nمثال على أوزان الانتباه (أول كلمتين):")
    for i, token in enumerate(attention.query_tokens[:2]):
        print(f"\n  '{token}':")
        top_3 = sorted(
            [(attention.concept_nodes[j], attention.weights[i][j])
             for j in range(len(attention.concept_nodes))],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        for node, weight in top_3:
            print(f"    {node:<15} {weight:.4f}")
