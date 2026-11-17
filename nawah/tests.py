# -*- coding: utf-8 -*-
"""
اختبارات شاملة لنظام نواة-1 (nawah/) ودمجه مع seshat (nawah_integration.py).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from token_analyzer import TokenAnalyzer
from attention import AttentionSystem, Softmax, SemanticEmbedding
from concepts import TripletExtractor, build_knowledge_graph
from novelty import NoveltyCalculator
from pipeline import NawahPipeline, IntentDetector
from nawah_integration import NawahEnhancedBrain, load_seshat_corpus


def run_tests():
    checks = 0

    # 1. TokenAnalyzer
    analyzer = TokenAnalyzer(corpus=["الذكاء الاصطناعي وتعلم الآلة والشبكات العصبية"])
    tokens = analyzer.analyze("كيف تتعلم الشبكات العصبية من البيانات؟")
    assert len(tokens) >= 4, "يجب تحليل كلمات السؤال"
    assert any(t.stem for t in tokens), "يجب استخراج الجذور"
    assert all(0.0 <= t.semantic_weight <= 1.0 for t in tokens), "الوزن الدلالي بين 0 و1"
    checks += 3

    # 2. Deterministic 16D Embeddings & Softmax safety
    emb = SemanticEmbedding(dim=16)
    v1 = emb.embed("سيشات")
    v2 = emb.embed("سيشات")
    assert v1 == v2 and len(v1) == 16, "التضمين الدلالي 16D يجب أن يكون حتميًا"
    norm_sq = sum(x * x for x in v1)
    assert abs(norm_sq - 1.0) < 1e-5, "المتجه يجب أن يكون مطبّعًا لطول 1"
    assert Softmax.compute([]) == [], "حماية Softmax من القائمة الفارغة"
    s = Softmax.compute([1.0, 2.0, 3.0])
    assert abs(sum(s) - 1.0) < 1e-6, "مجموع أوزان Softmax يساوي 1"
    checks += 4

    # 3. Self-Attention Q×Kᵀ/√d & empty guard
    att_sys = AttentionSystem(embedding_dim=16, num_heads=2)
    q_tokens = ["شبكة", "عصبية", "تعلم"]
    k_nodes = ["ذكاء", "بيانات", "تدريب", "نموذج"]
    matrix, ctx = att_sys.analyze_question(q_tokens, k_nodes)
    assert len(matrix.weights) == len(q_tokens), "صفوف الانتباه بعدد كلمات السؤال"
    assert len(matrix.weights[0]) == len(k_nodes), "أعمدة الانتباه بعدد العقد"
    assert len(ctx) == 16, "متجه السياق الموزون 16D"
    top_nodes = att_sys.get_relevant_nodes(matrix, top_k=2)
    assert len(top_nodes) == 2, "أهم عقدتين"
    empty_matrix, empty_ctx = att_sys.analyze_question([], [])
    assert empty_matrix.weights == [] and len(empty_ctx) == 16, "حماية من المدخلات الفارغة"
    checks += 5

    # 4. Concept Triplet Extraction & Knowledge Graph
    extractor = TripletExtractor()
    triplets = extractor.extract_from_text("الشبكات العصبية تعالج البيانات الضخمة. الذكاء الاصطناعي هو فرع حديث.")
    assert len(triplets) >= 1, "استخراج ثلاثية مفاهيمية واحدة على الأقل"
    kg = build_knowledge_graph("النموذج يتعلم الأنماط. الشبكة تحسب الأوزان.")
    assert len(kg.triplets) >= 1, "بناء الرسم البياني المعرفي"
    checks += 2

    # 5. Novelty Calculator
    nov = NoveltyCalculator(corpus=[{"content": "الذكاء الاصطناعي يحاكي القدرات الذهنية البشرية"}])
    res_copy = nov.calculate("الذكاء الاصطناعي يحاكي القدرات الذهنية البشرية", "ما هو الذكاء الاصطناعي؟")
    res_new = nov.calculate("تعتمد خوارزميات الانتشار الصورية على إزالة الضوضاء تدريجيًا عبر خطوات زمنية", "ما هو الذكاء الاصطناعي؟")
    assert 0.0 <= res_copy.novelty_score <= 1.0, "مقياس الأصالة في المدى الصحيح"
    assert res_new.novelty_score >= res_copy.novelty_score, "النص المبتكر أعلى أصالة من المنسوخ"
    checks += 2

    # 6. Intent Detector & NawahPipeline
    intent, label_ar, conf = IntentDetector.detect("ما هو خوفو؟")
    assert intent == "DEFINITION" and label_ar == "تعريف" and conf >= 0.8
    pipe = NawahPipeline(corpus=load_seshat_corpus(max_docs=30))
    inf = pipe.run("كيف يعمل نموذج سيشات؟", save_to_db=False)
    assert inf.generated_response and len(inf.analyzed_tokens) >= 2
    checks += 2

    # 7. NawahEnhancedBrain preserves valid Seshat answers and loads real corpus
    corpus = load_seshat_corpus(max_docs=40)
    assert len(corpus) >= 10, "يجب تحميل قاعدة معرفة seshat الفعلية من data/"
    brain = NawahEnhancedBrain(corpus=corpus)
    preserved = brain.enhance_reply("ما هي عاصمة مصر؟", "القاهرة هي عاصمة مصر.", is_fallback=False)
    assert preserved["reply"] == "القاهرة هي عاصمة مصر.", "لا يجوز استبدال إجابة seshat الصحيحة بقالب عام"
    assert "attention_matrix" in preserved["analysis"] and "analyzed_tokens" in preserved["analysis"]
    fallback_enhanced = brain.enhance_reply("ما هو الذكاء الاصطناعي؟", "", is_fallback=True)
    assert len(fallback_enhanced["reply"]) > 5, "توليد رد بديل عند غياب الرد الأساسي"
    checks += 4

    print(f"✅ نواة-1: كل الاختبارات نجحت ({checks} فحص)")


if __name__ == "__main__":
    run_tests()
