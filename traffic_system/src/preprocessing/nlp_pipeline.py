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
        
    def process_batch(self, batch_texts: list) -> torch.Tensor:
        # 1. Translation Step
        mt_tokens = self.mt_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        # Tweets are short; constrain generation so CPU processing remains practical.
        with torch.inference_mode():
            translated_ids = self.mt_model.generate(**mt_tokens, max_new_tokens=64)
        translated_texts = self.mt_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        
        # 2. Embedding Step
        bert_tokens = self.bert_tokenizer(translated_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.inference_mode():
            outputs = self.bert_model(**bert_tokens)
            
        # Masked mean pooling prevents padding tokens from affecting embeddings.
        attention_mask = bert_tokens["attention_mask"].unsqueeze(-1)
        summed = (outputs.last_hidden_state * attention_mask).sum(dim=1)
        return summed / attention_mask.sum(dim=1).clamp(min=1)
