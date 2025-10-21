"""
Orwellian Text Simplification + Clarity Feature Extractor
Purpose: Simplify complex legal sentences and extract linguistic clarity metrics
"""
import re
import nltk
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

# Download required NLTK data
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
        """Initialize the Orwellian simplifier"""
        pass
    
    def simplify_text(self, text):
        """
        Simplify complex legal text
        Args:
            text: string or list of strings
        Returns:
            simplified text in same format
        """
        if isinstance(text, list):
            return [self._simplify_single(t) for t in text]
        return self._simplify_single(text)
    
    def _simplify_single(self, text):
        """Simplify a single text string"""
        if not text:
            return text
        
        simplified = text
        
        # Rule 1: Replace legal jargon with simpler terms
        replacements = {
            r'\bhereinafter\b': 'from now on',
            r'\bwherein\b': 'where',
            r'\btherefore\b': 'so',
            r'\bfurthermore\b': 'also',
            r'\bnotwithstanding\b': 'despite',
            r'\bpursuant to\b': 'according to',
            r'\bprior to\b': 'before',
            r'\bsubsequent to\b': 'after',
            r'\bcommenced\b': 'started',
            r'\bterminated\b': 'ended',
            r'\bascertain\b': 'find out',
            r'\bapprehend\b': 'arrest',
            r'\babscond\b': 'run away',
        }
        
        for pattern, replacement in replacements.items():
            simplified = re.sub(pattern, replacement, simplified, flags=re.IGNORECASE)
        
        # Rule 2: Break long sentences (>30 words) at conjunctions
        sentences = sent_tokenize(simplified)
        result_sentences = []
        
        for sent in sentences:
            words = word_tokenize(sent)
            if len(words) > 30:
                # Try to split at ', and' or ', but' or similar
                split_sent = re.split(r',\s+(and|but|however)\s+', sent)
                if len(split_sent) > 1:
                    # Reconstruct sentences
                    for i in range(0, len(split_sent), 2):
                        if i+1 < len(split_sent):
                            result_sentences.append(split_sent[i].strip() + '.')
                        else:
                            result_sentences.append(split_sent[i].strip())
                else:
                    result_sentences.append(sent)
            else:
                result_sentences.append(sent)
        
        simplified = ' '.join(result_sentences)
        
        # Rule 3: Convert passive to active voice (simple heuristic)
        # This is a simplified approach; full passive-to-active needs complex NLP
        simplified = re.sub(r'was (\w+ed) by', r'\\1', simplified)
        
        return simplified
    
    def extract_features(self, text):
        """
        Extract clarity metrics from text
        Args:
            text: string or list of strings
        Returns:
            numpy array of features: [avg_sentence_length, lexical_density, passive_ratio]
        """
        if isinstance(text, list):
            text = ' '.join(text)
        
        if not text or len(text.strip()) == 0:
            return [0.0, 0.0, 0.0]
        
        # Feature 1: Average sentence length
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            avg_sent_len = 0.0
        else:
            words = word_tokenize(text)
            avg_sent_len = len(words) / len(sentences)
        
        # Feature 2: Lexical density (content words / total words)
        words = word_tokenize(text)
        if len(words) == 0:
            lexical_density = 0.0
        else:
            pos_tags = pos_tag(words)
            content_words = [w for w, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ', 'RB'))]
            lexical_density = len(content_words) / len(words)
        
        # Feature 3: Passive voice ratio
        passive_ratio = self._calculate_passive_ratio(text)
        
        return [avg_sent_len, lexical_density, passive_ratio]
    
    def _calculate_passive_ratio(self, text):
        """Calculate ratio of passive voice constructions"""
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            return 0.0
        
        passive_count = 0
        # Simple heuristic: look for "was/were/is/are + past participle"
        passive_pattern = r'\b(was|were|is|are|been)\s+\w+(ed|en)\b'
        
        for sent in sentences:
            if re.search(passive_pattern, sent, re.IGNORECASE):
                passive_count += 1
        
        return passive_count / len(sentences)
    
    def process_document(self, text):
        """
        Process a document: simplify and extract features
        Args:
            text: string or list of strings
        Returns:
            dict with 'simplified_text' and 'orwell_features'
        """
        simplified = self.simplify_text(text)
        features = self.extract_features(text)  # Extract from original
        
        return {
            'simplified_text': simplified,
            'orwell_features': features
        }