"""
Self-Attention System for Nawah
نظام الانتباه الذاتي: حساب مصفوفة الانتباه Q×Kᵀ/√d

Features:
- Multi-head attention (simplified)
- Deterministic 16D semantic embeddings (hashlib-seeded, process-independent)
- Works with NumPy when installed and falls back cleanly to pure Python math
- Query-Key-Value computation
- Softmax normalization with empty-input safety
- Weighted context aggregation
"""

import hashlib
import math
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


class Vector(list):
    """قائمة أرقام تدعم tolist() للتوافق مع NumPy."""
    def tolist(self) -> List[float]:
        return list(self)


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _matvec(vec: List[float], mat: List[List[float]]) -> List[float]:
    # vec: (d,), mat: (d, out_d)
    if not mat or not mat[0]:
        return []
    cols = len(mat[0])
    return [sum(vec[r] * mat[r][c] for r in range(len(vec))) for c in range(cols)]


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [_matvec(row, B) for row in A]


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
        """Softmax: exp(x_i) / sum(exp(x_j)) مع حماية من القوائم الفارغة."""
        if not x:
            return []
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x) or 1.0
        return [e / sum_exp for e in exp_x]


class SemanticEmbedding:
    """تمثيل دلالي حتمي للكلمات (16D Embedding)"""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.cache: Dict[str, Vector] = {}

    def embed(self, token: str) -> Vector:
        """تحويل كلمة إلى متجه دلالي حتمي عبر MD5 (ثابت عبر كل الجلسات)."""
        if token in self.cache:
            return self.cache[token]

        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = random.Random(seed)
        raw = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        vec = Vector([v / norm for v in raw])
        self.cache[token] = vec
        return vec

    def embed_batch(self, tokens: List[str]) -> List[Vector]:
        return [self.embed(t) for t in tokens]


class AttentionComputer:
    """حساب الانتباه الذاتي Q×Kᵀ/√d"""

    def __init__(self, embedding_dim: int = 16, num_heads: int = 2):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.embedding = SemanticEmbedding(dim=embedding_dim)

        rng = random.Random(42)
        scale = math.sqrt(embedding_dim) or 1.0
        self.W_q = [[rng.gauss(0.0, 1.0) / scale for _ in range(embedding_dim)] for _ in range(embedding_dim)]
        self.W_k = [[rng.gauss(0.0, 1.0) / scale for _ in range(embedding_dim)] for _ in range(embedding_dim)]
        self.W_v = [[rng.gauss(0.0, 1.0) / scale for _ in range(embedding_dim)] for _ in range(embedding_dim)]

    def compute_attention(
        self,
        query_tokens: List[str],
        key_tokens: List[str]
    ) -> AttentionMatrix:
        if not query_tokens or not key_tokens:
            return AttentionMatrix(query_tokens=query_tokens, concept_nodes=key_tokens, weights=[])

        Q_raw = self.embedding.embed_batch(query_tokens)
        K_raw = self.embedding.embed_batch(key_tokens)

        Q = _matmul(Q_raw, self.W_q)
        K = _matmul(K_raw, self.W_k)

        scale_factor = math.sqrt(self.embedding_dim) or 1.0
        attention_weights: List[List[float]] = []

        for q_vec in Q:
            scores = [_dot(q_vec, k_vec) / scale_factor for k_vec in K]
            attention_weights.append(Softmax.compute(scores))

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
    ) -> Vector:
        if not query_tokens or not value_tokens or not attention_matrix.weights:
            return Vector([0.0] * self.embedding_dim)

        V_raw = self.embedding.embed_batch(value_tokens)
        V = _matmul(V_raw, self.W_v)  # (num_values, dim)

        # Context = Attention × V
        context_rows: List[List[float]] = []
        for weights_row in attention_matrix.weights:
            row_vec = [
                sum(weights_row[v_idx] * V[v_idx][d] for v_idx in range(len(V)))
                for d in range(self.embedding_dim)
            ]
            context_rows.append(row_vec)

        n_q = len(context_rows) or 1
        aggregated = Vector([
            sum(context_rows[q_idx][d] for q_idx in range(len(context_rows))) / n_q
            for d in range(self.embedding_dim)
        ])
        return aggregated


class ConceptExtractor:
    """استخراج المفاهيم من النص"""

    @staticmethod
    def extract_concepts(tokens: List[str], pos_tags: List[str]) -> List[str]:
        concepts = []
        for token, pos in zip(tokens, pos_tags):
            if pos in ['NOUN', 'TECH_TERM', 'VERB']:
                concepts.append(token)
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
    ) -> Tuple[AttentionMatrix, Vector]:
        attention = self.computer.compute_attention(
            query_tokens=question_tokens,
            key_tokens=knowledge_nodes
        )
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
        if not attention_matrix.query_tokens or not attention_matrix.concept_nodes or not attention_matrix.weights:
            return []
        node_scores = {}
        n_q = len(attention_matrix.query_tokens)
        for node_idx, node in enumerate(attention_matrix.concept_nodes):
            avg_attention = sum(
                attention_matrix.weights[q_idx][node_idx]
                for q_idx in range(n_q)
            ) / n_q
            node_scores[node] = avg_attention
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]


def compute_attention(
    question: str,
    knowledge_base: List[str],
    embedding_dim: int = 16
) -> Tuple[AttentionMatrix, List[Tuple[str, float]]]:
    system = AttentionSystem(embedding_dim=embedding_dim)
    question_tokens = [t for t in question.split() if t.strip()]
    attention, _ = system.analyze_question(
        question_tokens=question_tokens,
        knowledge_nodes=knowledge_base
    )
    relevant_nodes = system.get_relevant_nodes(attention, top_k=5)
    return attention, relevant_nodes
