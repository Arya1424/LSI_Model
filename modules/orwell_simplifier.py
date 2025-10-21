"""
Orwellian Text Simplification + Clarity Feature Extractor
Purpose: Extract linguistic clarity metrics (features) for model input.
"""
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from typing import List

# NLTK data check (assumes NLTK is installed)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

class OrwellSimplifier:
    def __init__(self):
        pass
    
    def extract_features(self, text: str) -> List[float]:
        """
        Extract clarity metrics: [avg_sent_len, lexical_density, passive_ratio]
        """
        if not text or len(text.strip()) == 0:
            return [0.0, 0.0, 0.0]
        
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if len(words) == 0:
            return [0.0, 0.0, 0.0]
        
        # Feature 1: Average sentence length
        avg_sent_len = (len(words) / len(sentences)) if len(sentences) > 0 else 0.0
        
        # Feature 2: Lexical density
        pos_tags = pos_tag(words)
        content_words = [w for w, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ', 'RB'))]
        lexical_density = len(content_words) / len(words)
        
        # Feature 3: Passive voice ratio
        passive_ratio = self._calculate_passive_ratio(sentences)
        
        return [avg_sent_len, lexical_density, passive_ratio]
    
    def _calculate_passive_ratio(self, sentences: List[str]) -> float:
        """Calculate ratio of passive voice constructions using a simple heuristic."""
        if len(sentences) == 0:
            return 0.0
        
        passive_count = 0
        # Simple heuristic: look for "is/was/were/are/been + VBN/VBD"
        passive_pattern = re.compile(r'\b(is|was|were|are|been)\b\s+\w+(ed|en)\b', re.IGNORECASE)
        
        for sent in sentences:
            if passive_pattern.search(sent):
                passive_count += 1
        
        return passive_count / len(sentences)