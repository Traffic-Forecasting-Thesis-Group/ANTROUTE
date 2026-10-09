"""
===============================================================================
ARCHITECTURE BLOCKS 3, 4 & 5: MarianMT -> Tokenizer -> DistilBERT
===============================================================================
Diagram path:   Textual Pre-cleaning (block 2)
                  -> [MarianMT]    Tagalog-English translation
                  -> [TOKENIZER]
                  -> [DISTILBERT]  text embedding
                  -> CNN + LSTM (block 9)

NOTE FOR READERS
    The diagram draws these as three separate boxes. In code they are three
    stages of one class, BatchNLPPipeline, because they always run together on
    the same batch and share a device. Follow process_batch() top to bottom and
    you are walking those three boxes in order.

WHAT IT DOES
    Turns a cleaned tweet into one 768-dimensional vector that represents its
    meaning, which is the only form the neural network can consume.

      Block 3  MarianMT (Helsinki-NLP/opus-mt-tl-en) translates Tagalog to
               English. It is Tagalog->English ONLY, so the pipeline first
               detects language and skips text already in English -- feeding
               English in produces garbled output (see _is_english).
      Block 4  The DistilBERT tokenizer splits text into sub-word tokens and
               pads/truncates the batch to a uniform 512 tokens.
      Block 5  DistilBERT (distilbert-base-uncased) encodes those tokens, and
               masked mean pooling collapses them into ONE 768-d vector per
               post. Masking matters: without it, padding would drag every
               short tweet's embedding toward the same value.

INPUT   <- cleaned_text from TextCleaner (block 2)
OUTPUT  -> 768-d embedding per post. Collected into
           data/processed/embeddings.pt, which CNNLSTMFusion (block 9) reads
           as its text branch.

KEY NAMES
    _is_english()    guards MarianMT against English input
    process_batch()  blocks 3 -> 4 -> 5 in one pass; numbered steps inside
===============================================================================
"""

import torch
from transformers import MarianMTModel, MarianTokenizer, DistilBertModel, DistilBertTokenizer
from langdetect import detect, LangDetectException, DetectorFactory

# langdetect uses random sampling internally for short/ambiguous text;
# pinning the seed makes results reproducible across runs and machines.
DetectorFactory.seed = 0


class BatchNLPPipeline:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # [BLOCK 3] MarianMT Tagalog to English Translation
        mt_model_name = "Helsinki-NLP/opus-mt-tl-en"
        self.mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
        self.mt_model = MarianMTModel.from_pretrained(mt_model_name).to(self.device).eval()

        # [BLOCKS 4 & 5] DistilBERT tokenizer + text embedding model
        bert_model_name = "distilbert-base-uncased"
        self.bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = DistilBertModel.from_pretrained(bert_model_name).to(self.device).eval()

    @staticmethod
    def _is_english(text: str) -> bool:
        """
        Detects whether a text is already English, so we can skip MarianMT
        translation for it. MarianMT (opus-mt-tl-en) is Tagalog->English only;
        feeding it text that's already English produces garbled output
        (e.g. "recallwas injured his motorcycle fire choughing a coldlisionwo").

        Defaults to English (skip translation) on empty text or detection
        failure, since that's the safer fallback: an unnecessary translation
        attempt on genuinely Tagalog text would still fail loudly at the
        embedding step, whereas mis-skipping is a silent quality hit.
        """
        if not text or not text.strip():
            return True
        try:
            return detect(text) == "en"
        except LangDetectException:
            return True

    def process_batch(self, batch_texts: list[str]) -> list[dict]:
        """
        Translates non-English batch items to English (skipping items already
        in English) and generates DistilBERT embeddings for all items.
        Returns a list of dicts with {"english_translation": str, "embedding_vector": list[float]}
        """
        if not batch_texts:
            return []

        # 0. Language detection: split the batch into items that need
        # translation vs. items that are already English.
        is_english_flags = [self._is_english(text) for text in batch_texts]
        translate_idx = [i for i, is_eng in enumerate(is_english_flags) if not is_eng]

        # translated_texts is aligned 1:1 with batch_texts by index.
        translated_texts: list[str] = list(batch_texts)  # default: pass-through for English items

        if translate_idx:
            texts_to_translate = [batch_texts[i] for i in translate_idx]

            # 1. [BLOCK 3] Translation Step (only for the non-English subset)
            # Max length padding for Tweets/News up to 512 tokens
            mt_tokens = self.mt_tokenizer(
                texts_to_translate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.inference_mode():
                # max_new_tokens increased to handle longer texts (e.g. news snippets)
                translated_ids = self.mt_model.generate(**mt_tokens, max_new_tokens=128)

            translated_subset = self.mt_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)

            for i, translated in zip(translate_idx, translated_subset):
                translated_texts[i] = translated

        # 2. [BLOCK 4] Tokenizer Step: sub-word tokens, padded/truncated to 512.
        # Runs on all items: translated non-English + pass-through English.
        bert_tokens = self.bert_tokenizer(
            translated_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.bert_model(**bert_tokens)

        # 3. [BLOCK 5] DistilBERT text embedding. Masked mean pooling collapses
        # the per-token outputs into one 768-d vector; the mask stops padding
        # from pulling every short tweet toward the same embedding.
        attention_mask = bert_tokens["attention_mask"].unsqueeze(-1)
        summed = (outputs.last_hidden_state * attention_mask).sum(dim=1)
        embeddings = summed / attention_mask.sum(dim=1).clamp(min=1)

        # Move embeddings back to CPU and convert to standard python lists
        embeddings_list = embeddings.cpu().numpy().tolist()

        results = []
        for eng_text, emb_vec in zip(translated_texts, embeddings_list):
            results.append({
                "english_translation": eng_text,
                "embedding_vector": emb_vec
            })

        return results