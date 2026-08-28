import json
from torch.utils.data import Dataset

class TwitterTrafficDataset(Dataset):
    def __init__(self, filepath: str, cleaner):
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_content = json.load(f)
            # Adjust the key based on the actual twitterapi.io JSON response structure
            self.data = raw_content.get('data', raw_content) if isinstance(raw_content, dict) else raw_content
        self.cleaner = cleaner

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Safely extract text, defaulting to an empty string if missing
        tweet_obj = self.data[idx]
        raw_tweet = tweet_obj.get('text', '') if isinstance(tweet_obj, dict) else str(tweet_obj)
        
        return self.cleaner.clean(raw_tweet)