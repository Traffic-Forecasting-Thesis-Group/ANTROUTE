import torch
from transformers import MarianMTModel, MarianTokenizer, DistilBertModel, DistilBertTokenizer

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
        
    def process_batch(self, batch_texts: list[str]) -> list[dict]:
        """
        Translates Tagalog batch to English and generates DistilBERT embeddings.
        Returns a list of dicts with {"english_translation": str, "embedding_vector": list[float]}
        """
        if not batch_texts:
            return []

        # 1. Translation Step
        # Max length padding for Tweets/News up to 512 tokens
        mt_tokens = self.mt_tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.inference_mode():
            # max_new_tokens increased to handle longer texts (e.g. news snippets)
            translated_ids = self.mt_model.generate(**mt_tokens, max_new_tokens=128)

        translated_texts = self.mt_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        
        # 2. Embedding Step
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

