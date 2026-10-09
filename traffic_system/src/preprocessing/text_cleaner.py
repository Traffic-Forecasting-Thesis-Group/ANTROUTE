"""
===============================================================================
ARCHITECTURE BLOCK 2: Textual Pre-cleaning
===============================================================================
Diagram path:   Unstructured Data (X/Twitter)  ->  [TEXTUAL PRE-CLEANING]
                                               ->  MarianMT (block 3)

WHAT IT DOES
    Prepares a raw tweet for translation. Three jobs, in this order:
      1. Strip the noise MarianMT would otherwise try to translate:
         URLs, @mentions, and runs of extra whitespace.
      2. Rewrite Tagalog traffic slang that MarianMT has no vocabulary for
         ("buhol-buhol" -> "heavy traffic"). Without this the translator
         emits garbage for exactly the posts we care most about.
      3. Find and preserve location anchors (EDSA, Ortigas, Roxas Blvd...).
         These survive translation intact and are what later lets a tweet be
         attached to a specific camera intersection.

    Why step 3 matters downstream: a tweet that names no monitored junction
    is dropped at training time (src/data/training_data.py), because giving
    every camera the same tweet makes the text feature identical everywhere
    and therefore useless.

INPUT   <- raw tweet text, from src/ingestion/twitter_scraper.py (block 1)
OUTPUT  -> {"cleaned_text", "entities"}, consumed by BatchNLPPipeline (block 3)

KEY NAMES
    remove_noise()                  URLs / @mentions / whitespace
    replace_slang()                 the MarianMT out-of-vocabulary fix
    extract_and_preserve_entities() the location anchors
    clean()                         runs all three in order
===============================================================================
"""

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
                "EDSA", "EDSA Guadalupe", "EDSA Ortigas", "EDSA Cubao",
                "EDSA Kamuning", "EDSA Balintawak", "EDSA Taft",
                "Roxas Boulevard", "Roxas Blvd", "Makati", "Quezon City",
                "Manila", "Metro Manila", "Pasay", "Ortigas", "Pasig",
                "Novaliches", "Navotas", "Mandaluyong", "Paranaque",
                "Las Pinas", "Muntinlupa", "San Juan", "Taguig", "Marikina",
                "Caloocan", "Malabon", "Philippines", "Pilipinas", "C5",
                "NLEX", "SLEX", "Skyway"
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
            
    # [BLOCK 2] "Removes URLs, @mentions, extra whitespace" on the diagram.
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

    # [BLOCK 2] "preserves key entities" on the diagram. These location anchors
    # are how a tweet later gets attached to one camera's intersection.
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

    # [BLOCK 2] Entry point: the whole pre-cleaning block in one call.
    def clean(self, raw_text: str) -> dict:
        denoised = self.remove_noise(raw_text)
        de_slanged = self.replace_slang(denoised)
        cleaned_text, entities = self.extract_and_preserve_entities(de_slanged)

        return {
            "cleaned_text": cleaned_text,
            "entities": entities
        }