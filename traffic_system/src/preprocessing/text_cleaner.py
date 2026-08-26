import re
import json

class TextCleaner:
    def __init__(self, gazetteer_path: str):
        with open(gazetteer_path, 'r', encoding='utf-8') as f:
            self.locations = json.load(f)
            
    def remove_noise(self, text: str) -> str:
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def preserve_entities(self, text: str) -> str:
        for loc in self.locations:
            pattern = re.compile(re.escape(loc), re.IGNORECASE)
            text = pattern.sub(loc, text)
        return text

    def clean(self, raw_tweet: str) -> str:
        denoised = self.remove_noise(raw_tweet)
        return self.preserve_entities(denoised)