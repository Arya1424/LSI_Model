"""
Trie-based Legal Phrase Annotator
Purpose: Normalize and tag known legal phrases or IPC section names.
"""
import pygtrie
import re
from typing import Dict, List, Union

class TrieAnnotator:
    def __init__(self, phrases_dict: Dict[str, str] = None):
        """Initialize Trie with legal phrases."""
        self.trie = pygtrie.CharTrie()
        if phrases_dict is None:
            phrases_dict = self._build_default_ipc_phrases()
        for phrase, tag in phrases_dict.items():
            self.trie[phrase.lower()] = tag.upper() 
    
    def _build_default_ipc_phrases(self) -> Dict[str, str]:
        """Build default IPC section phrases"""
        return {
            'attempt to murder': '<IPC_307>',
            'culpable homicide': '<IPC_299>',
            'murder': '<IPC_300>',
            'voluntarily causing hurt': '<IPC_321>',
            'voluntarily causing hurt on provocation': '<IPC_334>',
            'robbery': '<IPC_390>',
            'dacoity with murder': '<IPC_396>',
            'criminal trespass': '<IPC_441>',
            'house breaking by night': '<IPC_446>',
            'theft': '<IPC_378>',
            'criminal intimidation': '<IPC_506>',
            'criminal conspiracy': '<IPC_120B>',
            'wrongful restraint': '<IPC_339>',
            'wrongful confinement': '<IPC_342>',
            'kidnapping': '<IPC_363>',
            'abduction': '<IPC_362>',
            'rape': '<IPC_376>',
            'outraging modesty': '<IPC_354>',
            'dowry death': '<IPC_304B>',
            'cruelty by husband': '<IPC_498A>',
            'cheating': '<IPC_420>',
            'forgery': '<IPC_463>',
            'breach of trust': '<IPC_406>',
            'criminal breach of trust': '<IPC_405>',
        }
    
    def annotate_text(self, text: str) -> str:
        """Annotate a single text string by replacing legal phrases with tags."""
        if not text: return text
        
        text_lower = text.lower()
        words = text_lower.split()
        replacements = []
        
        for i in range(len(words)):
            for j in range(i + 1, min(i + 6, len(words) + 1)):
                phrase = ' '.join(words[i:j])
                match = self.trie.longest_prefix(phrase)
                if match and match.key == phrase:
                    replacements.append((i, len(match.key.split()), match.value))

        replacements.sort(key=lambda x: (x[0], -x[1]))
        
        new_words = words[:]
        used_token_indices = set()
        
        for start_idx, length, tag in replacements:
            end_idx = start_idx + length
            is_overlap = any(i in used_token_indices for i in range(start_idx, end_idx))
            
            if not is_overlap:
                new_words[start_idx] = tag 
                for i in range(start_idx + 1, end_idx):
                    new_words[i] = ''
                for i in range(start_idx, end_idx):
                    used_token_indices.add(i)

        annotated_text = ' '.join(filter(None, new_words))
        return annotated_text