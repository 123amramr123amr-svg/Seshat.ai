"""
Nawah - نواة
البنية التحتية لنموذج الذكاء الاصطناعي التوليدي من الصفر

Features:
- Token Analysis (تحليل الكلمات)
- Self-Attention (نظام الانتباه)
- Concept Extraction (استخراج المفاهيم)
- Novelty Scoring (قياس الأصالة)
- SQLite Persistence (قاعدة بيانات)
"""

__version__ = '1.0.0'
__author__ = 'Amr Ahmed'

from .token_analyzer import TokenAnalyzer, analyze_text, AnalyzedToken
from .attention import AttentionSystem, compute_attention, AttentionMatrix
from .concepts import TripletExtractor, build_knowledge_graph, ConceptTriplet
from .novelty import NoveltyCalculator, calculate_novelty, NoveltyResult
from .database import NawahDatabase, get_db

__all__ = [
    # Token Analysis
    'TokenAnalyzer',
    'analyze_text',
    'AnalyzedToken',
    
    # Attention
    'AttentionSystem',
    'compute_attention',
    'AttentionMatrix',
    
    # Concepts
    'TripletExtractor',
    'build_knowledge_graph',
    'ConceptTriplet',
    
    # Novelty
    'NoveltyCalculator',
    'calculate_novelty',
    'NoveltyResult',
    
    # Database
    'NawahDatabase',
    'get_db',
]
