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

        # MarianMT Tagalog to English Translation
        mt_model_name = "Helsinki-NLP/opus-mt-tl-en"
        self.mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
        self.mt_model = MarianMTModel.from_pretrained(mt_model_name).to(self.device).eval()

        # DistilBERT Text Embedding
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

            # 1. Translation Step (only for the non-English subset)
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

        # 2. Embedding Step (runs on all items: translated non-English + pass-through English)
        bert_tokens = self.bert_tokenizer(
            translated_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.bert_model(**bert_tokens)

        # Masked mean pooling prevents padding tokens from affecting embeddings.
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