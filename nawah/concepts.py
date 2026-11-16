"""
Concept Triplet Extraction for Nawah
استخراج المفاهيم الثلاثية: فاعل-فعل-مفعول (Subject-Predicate-Object)

Features:
- Extract semantic triplets from Arabic text
- Identify subject, predicate, and object
- Build knowledge graph from text
- Support for Arabic grammar patterns
"""

import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ConceptTriplet:
    """ثلاثية مفاهيمية: فاعل-فعل-مفعول"""
    subject: str      # الفاعل
    predicate: str    # الفعل/العلاقة
    object: str       # المفعول
    confidence: float # الثقة (0-1)
    
    def to_dict(self) -> Dict:
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'confidence': self.confidence
        }
    
    def to_string(self) -> str:
        return f"({self.subject} ← {self.predicate} → {self.object})"


class ArabicGrammarParser:
    """محلل قواعد اللغة العربية (مبسط)"""
    
    # أنماط الجمل العربية
    VERBAL_SENTENCE = re.compile(r'^(ي|ت|ا|ن)?(\w+)(ون|ين|ات|ان)?\s+(\w+)\s+(\w+)')
    NOMINAL_SENTENCE = re.compile(r'^(\w+)\s+(ي|ت|ا|ن)?(\w+)')
    
    # أدوات الربط
    CONJUNCTIONS = {'و', 'أو', 'ثم', 'لكن', 'بل', 'حتى', 'كي'}
    
    # حروف الجر
    PREPOSITIONS = {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بين', 'عند', 'لدى', 'حول'}
    
    # ضمائر
    PRONOUNS = {'هو', 'هي', 'هم', 'هن', 'أنا', 'نحن', 'أنت', 'أنتم'}
    
    @classmethod
    def is_verb(cls, word: str) -> bool:
        """هل الكلمة فعل؟"""
        # أفعال مضارعة
        if word.startswith(('ي', 'ت', 'ا', 'ن')) and len(word) > 3:
            return True
        
        # أفعال ماضية شائعة
        past_verbs = ['كان', 'قال', 'فعل', 'عمل', 'كتب', 'قرأ', 'فهم', 'سأل', 'رد', 'بنى']
        if word in past_verbs:
            return True
        
        return False
    
    @classmethod
    def is_noun(cls, word: str) -> bool:
        """هل الكلمة اسم؟"""
        # الأسماء تبدأ بـ "ال"
        if word.startswith('ال') and len(word) > 3:
            return True
        
        # أسماء شائعة
        common_nouns = ['نموذج', 'ذكاء', 'شبكة', 'بيانات', 'سؤال', 'رد', 'جملة', 'كلمة']
        if word in common_nouns:
            return True
        
        return False


class TripletExtractor:
    """استخراج الثلاثيات المفاهيمية"""
    
    def __init__(self):
        self.parser = ArabicGrammarParser()
    
    def extract_from_sentence(self, sentence: str) -> List[ConceptTriplet]:
        """
        استخراج الثلاثيات من جملة واحدة
        
        Args:
            sentence: الجملة العربية
        
        Returns:
            قائمة ConceptTriplet
        """
        # تنظيف الجملة
        sentence = self._clean_sentence(sentence)
        tokens = sentence.split()
        
        if len(tokens) < 2:
            return []
        
        triplets = []
        
        # محاولة 1: جملة فعلية (فعل-فاعل-مفعول)
        verbal_triplet = self._extract_verbal_triplet(tokens)
        if verbal_triplet:
            triplets.append(verbal_triplet)
        
        # محاولة 2: جملة اسمية (مبتدأ-خبر)
        nominal_triplet = self._extract_nominal_triplet(tokens)
        if nominal_triplet:
            triplets.append(nominal_triplet)
        
        # محاولة 3: علاقات بحروف الجر
        prep_triplets = self._extract_preposition_triplets(tokens)
        triplets.extend(prep_triplets)
        
        return triplets
    
    def _clean_sentence(self, sentence: str) -> str:
        """تنظيف الجملة"""
        # إزالة علامات الترقيم
        sentence = re.sub(r'[؟?!.،,]', '', sentence)
        # إزالة التشكيل
        sentence = re.sub(r'[\u064B-\u0652]', '', sentence)
        return sentence.strip()
    
    def _extract_verbal_triplet(self, tokens: List[str]) -> Optional[ConceptTriplet]:
        """استخراج ثلاثية من جملة فعلية"""
        # ابحث عن فعل
        verb_idx = None
        for i, token in enumerate(tokens):
            if self.parser.is_verb(token):
                verb_idx = i
                break
        
        if verb_idx is None:
            return None
        
        # الفعل هو العلاقة
        predicate = tokens[verb_idx]
        
        # الفاعل بعد الفعل
        if verb_idx + 1 < len(tokens):
            subject = tokens[verb_idx + 1]
        else:
            return None
        
        # المفعول بعد الفاعل
        if verb_idx + 2 < len(tokens):
            obj = tokens[verb_idx + 2]
        else:
            # إذا مفيش مفعول، نستخدم الفاعل كـ subject و object
            obj = subject
        
        return ConceptTriplet(
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence=0.8
        )
    
    def _extract_nominal_triplet(self, tokens: List[str]) -> Optional[ConceptTriplet]:
        """استخراج ثلاثية من جملة اسمية (مبتدأ-خبر)"""
        if len(tokens) < 2:
            return None
        
        # المبتدأ أول كلمة (اسم)
        if self.parser.is_noun(tokens[0]):
            subject = tokens[0]
            
            # الخبر ثاني كلمة
            predicate = 'هو'  # علاقة "هو"
            obj = tokens[1]
            
            return ConceptTriplet(
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence=0.7
            )
        
        return None
    
    def _extract_preposition_triplets(self, tokens: List[str]) -> List[ConceptTriplet]:
        """استخراج ثلاثيات من علاقات حروف الجر"""
        triplets = []
        
        for i, token in enumerate(tokens):
            if token in self.parser.PREPOSITIONS:
                # الحرف هو العلاقة
                predicate = token
                
                # الاسم قبل الحرف (إن وجد)
                if i > 0:
                    subject = tokens[i - 1]
                else:
                    continue
                
                # الاسم بعد الحرف
                if i + 1 < len(tokens):
                    obj = tokens[i + 1]
                    
                    triplets.append(ConceptTriplet(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence=0.6
                    ))
        
        return triplets
    
    def extract_from_text(self, text: str) -> List[ConceptTriplet]:
        """
        استخراج الثلاثيات من نص كامل
        
        Args:
            text: النص العربي
        
        Returns:
            قائمة ConceptTriplet
        """
        # تقسيم النص إلى جمل
        sentences = re.split(r'[.؟!]', text)
        
        all_triplets = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                triplets = self.extract_from_sentence(sentence)
                all_triplets.extend(triplets)
        
        # إزالة التكرار
        unique_triplets = []
        seen = set()
        for t in all_triplets:
            key = (t.subject, t.predicate, t.object)
            if key not in seen:
                seen.add(key)
                unique_triplets.append(t)
        
        return unique_triplets


class KnowledgeGraph:
    """رسم بياني معرفي من الثلاثيات"""
    
    def __init__(self):
        self.triplets: List[ConceptTriplet] = []
        self.nodes: Dict[str, List[Tuple[str, str]]] = {}  # node -> [(relation, target)]
    
    def add_triplet(self, triplet: ConceptTriplet):
        """إضافة ثلاثية للرسم البياني"""
        self.triplets.append(triplet)
        
        # إضافة للعقدة المصدر
        if triplet.subject not in self.nodes:
            self.nodes[triplet.subject] = []
        self.nodes[triplet.subject].append((triplet.predicate, triplet.object))
    
    def add_triplets(self, triplets: List[ConceptTriplet]):
        """إضافة عدة ثلاثيات"""
        for t in triplets:
            self.add_triplet(t)
    
    def get_related(self, node: str, max_depth: int = 1) -> List[Tuple[str, str, int]]:
        """
        جلب العقد المرتبطة بعقدة معينة
        
        Args:
            node: العقدة المطلوبة
            max_depth: أقصى عمق للبحث
        
        Returns:
            قائمة (relation, target, depth)
        """
        related = []
        visited = set()
        
        def traverse(current_node: str, depth: int):
            if depth > max_depth or current_node in visited:
                return
            
            visited.add(current_node)
            
            if current_node in self.nodes:
                for relation, target in self.nodes[current_node]:
                    related.append((relation, target, depth))
                    traverse(target, depth + 1)
        
        traverse(node, 0)
        return related
    
    def to_dict(self) -> Dict:
        """تحويل الرسم البياني إلى قاموس"""
        return {
            'nodes': list(self.nodes.keys()),
            'edges': [
                {
                    'subject': t.subject,
                    'predicate': t.predicate,
                    'object': t.object,
                    'confidence': t.confidence
                }
                for t in self.triplets
            ],
            'num_nodes': len(self.nodes),
            'num_edges': len(self.triplets)
        }


# Convenience function
def extract_concepts(text: str) -> List[ConceptTriplet]:
    """
    دالة مساعدة لاستخراج المفاهيم من نص
    
    Args:
        text: النص العربي
    
    Returns:
        قائمة ConceptTriplet
    """
    extractor = TripletExtractor()
    return extractor.extract_from_text(text)


def build_knowledge_graph(text: str) -> KnowledgeGraph:
    """
    بناء رسم بياني معرفي من نص
    
    Args:
        text: النص العربي
    
    Returns:
        KnowledgeGraph
    """
    extractor = TripletExtractor()
    triplets = extractor.extract_from_text(text)
    
    graph = KnowledgeGraph()
    graph.add_triplets(triplets)
    
    return graph


# Test
if __name__ == '__main__':
    # مثال على الاستخدام
    sample_texts = [
        'النموذج يفهم الأسئلة',
        'الذكاء الاصطناعي يتعلم من البيانات',
        'الشبكة العصبية تحاكي الدماغ',
        'ازاي النموذج بيفهم الأسئلة؟',
    ]
    
    print("استخراج المفاهيم الثلاثية:\n")
    print("=" * 80)
    
    for text in sample_texts:
        print(f"\nالنص: {text}")
        triplets = extract_concepts(text)
        
        if triplets:
            print(f"الثلاثيات المستخرجة ({len(triplets)}):")
            for t in triplets:
                print(f"  {t.to_string()} (ثقة: {t.confidence:.2f})")
        else:
            print("  لم يتم استخراج ثلاثيات")
    
    print("\n" + "=" * 80)
    print("\nبناء رسم بياني معرفي:\n")
    
    full_text = """
    النموذج يفهم الأسئلة. الذكاء الاصطناعي يتعلم من البيانات.
    الشبكة العصبية تحاكي الدماغ البشري. النظام يولد ردود جديدة.
    """
    
    graph = build_knowledge_graph(full_text)
    
    print(f"الرسم البياني:")
    print(f"  عدد العقد: {graph.to_dict()['num_nodes']}")
    print(f"  عدد الحواف: {graph.to_dict()['num_edges']}")
    
    print(f"\nالعقد:")
    for node in list(graph.nodes.keys())[:10]:
        print(f"  {node}")
    
    print(f"\nمثال على العلاقات:")
    for triplet in graph.triplets[:5]:
        print(f"  {triplet.to_string()}")
    
    # البحث عن علاقات
    if 'النموذج' in graph.nodes:
        print(f"\nالعلاقات المرتبطة بـ 'النموذج':")
        related = graph.get_related('النموذج', max_depth=1)
        for relation, target, depth in related:
            print(f"  النموذج ← {relation} → {target} (عمق: {depth})")
