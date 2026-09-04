import re
import json

class TextCleaner:
    def __init__(self, gazetteer_path: str = None):
        if gazetteer_path:
            with open(gazetteer_path, 'r', encoding='utf-8') as f:
                self.locations = json.load(f)
        else:
            # Fallback gazetteer based on pipeline_context
            self.locations = [
                "EDSA", "Roxas Blvd", "Makati", "Quezon City", "Manila",
                "Pasay", "C5", "NLEX", "SLEX", "Skyway"
            ]

        # Address OOV slangs that trip up MarianMT
        self.slang_dict = {
            r'\bbuhol-?buhol\b': 'heavy traffic',
            r'\bkamote\s?rider\b': 'reckless motorcycle driver',
            r'\btrapik\b': 'traffic',
            r'\bbaha\b': 'flood',
            r'\bbumabaha\b': 'flooding',
            r'\bnabangga\b': 'crashed'
        }
            
    def remove_noise(self, text: str) -> str:
        # Strip URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        # Strip Mentions
        text = re.sub(r'@\w+', '', text)
        # Strip arbitrary extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def replace_slang(self, text: str) -> str:
        for pattern, replacement in self.slang_dict.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def extract_and_preserve_entities(self, text: str) -> tuple[str, list[str]]:
        entities_found = set()

        # We can also detect tags as entities
        tags = re.findall(r'#\w+', text)
        for t in tags:
            entities_found.add(t)

        for loc in self.locations:
            pattern = re.compile(r'\b' + re.escape(loc) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                entities_found.add(loc)
                # Capitalize nicely for preservation (NER help)
                text = pattern.sub(loc, text)

        return text, list(entities_found)

    def clean(self, raw_text: str) -> dict:
        denoised = self.remove_noise(raw_text)
        de_slanged = self.replace_slang(denoised)
        cleaned_text, entities = self.extract_and_preserve_entities(de_slanged)

        return {
            "cleaned_text": cleaned_text,
            "entities": entities
        }