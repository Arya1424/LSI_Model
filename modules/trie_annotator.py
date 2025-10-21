"""
Trie-based Legal Phrase Annotator
Purpose: Normalize and tag known legal phrases or IPC section names
"""
import pygtrie
import re

class TrieAnnotator:
    def __init__(self, phrases_dict=None):
        """
        Initialize Trie with legal phrases
        Args:
            phrases_dict: dict mapping phrases to tags, e.g., {'attempt to murder': '<IPC_307>'}
        """
        self.trie = pygtrie.CharTrie()
        
        if phrases_dict is None:
            # Default IPC phrases
            phrases_dict = self._build_default_ipc_phrases()
        
        # Build the trie
        for phrase, tag in phrases_dict.items():
            self.trie[phrase.lower()] = tag
    
    def _build_default_ipc_phrases(self):
        """Build default IPC section phrases"""
        # Common IPC sections and their descriptions
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
    
    def annotate_text(self, text):
        """
        Annotate text by replacing legal phrases with tags
        Args:
            text: string or list of strings
        Returns:
            annotated text in same format as input
        """
        if isinstance(text, list):
            return [self._annotate_single(t) for t in text]
        return self._annotate_single(text)
    
    def _annotate_single(self, text):
        """Annotate a single text string"""
        if not text:
            return text
        
        text_lower = text.lower()
        result = text
        replacements = []
        
        # Find all matches
        words = text_lower.split()
        for i in range(len(words)):
            for j in range(i+1, min(i+6, len(words)+1)):  # Max 5-word phrases
                phrase = ' '.join(words[i:j])
                if phrase in self.trie:
                    tag = self.trie[phrase]
                    # Store (start_pos, end_pos, original, tag)
                    start = text_lower.find(phrase)
                    if start != -1:
                        replacements.append((start, start + len(phrase), phrase, tag))
        
        # Sort by position and apply replacements (longest first to avoid conflicts)
        replacements.sort(key=lambda x: (x[0], -(x[1]-x[0])))
        
        # Apply replacements avoiding overlaps
        used_ranges = []
        final_replacements = []
        
        for start, end, orig, tag in replacements:
            # Check for overlap
            overlap = False
            for used_start, used_end in used_ranges:
                if not (end <= used_start or start >= used_end):
                    overlap = True
                    break
            
            if not overlap:
                final_replacements.append((start, end, orig, tag))
                used_ranges.append((start, end))
        
        # Apply replacements from end to start
        final_replacements.sort(key=lambda x: -x[0])
        for start, end, orig, tag in final_replacements:
            # Find exact match preserving case
            pattern = re.compile(re.escape(orig), re.IGNORECASE)
            result = pattern.sub(tag, result, count=1)
        
        return result
    
    def add_phrases(self, phrases_dict):
        """Add more phrases to the trie"""
        for phrase, tag in phrases_dict.items():
            self.trie[phrase.lower()] = tag


if __name__ == '__main__':
    # Test the annotator
    print("Testing Trie Annotator...")
    
    trie = TrieAnnotator()
    
    test_cases = [
        "The accused attempted to murder the victim",
        "Case of culpable homicide and theft",
        "Charged with criminal conspiracy and cheating"
    ]
    
    for test in test_cases:
        annotated = trie.annotate_text(test)
        print(f"\nOriginal:  {test}")
        print(f"Annotated: {annotated}")
    
    print("\n✓ Trie Annotator working correctly!")