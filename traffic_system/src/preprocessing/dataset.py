import json
from pathlib import Path
from torch.utils.data import Dataset

class TwitterTrafficDataset(Dataset):
    def __init__(self, filepath: str, cleaner):
        """Load one raw JSON file or every organized JSON file below a folder."""
        source = Path(filepath)
        paths = [source] if source.is_file() else sorted(source.rglob("tweets_*.json"))
        if not paths:
            raise FileNotFoundError(f"No tweet datasets found at {source}")

        self.data = []
        for path in paths:
            with path.open(encoding='utf-8') as file:
                raw_content = json.load(file)
            tweets = raw_content.get('data', raw_content) if isinstance(raw_content, dict) else raw_content
            if not isinstance(tweets, list):
                raise ValueError(f"Expected a list of tweets in {path}")
            self.data.extend(tweets)
        self.cleaner = cleaner

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Safely extract text, defaulting to an empty string if missing
        tweet_obj = self.data[idx]
        raw_tweet = tweet_obj.get('text', '') if isinstance(tweet_obj, dict) else str(tweet_obj)

        # Return only the cleaned text string (not the full dict with "entities").
        # DataLoader's default collate_fn can batch a list of strings directly,
        # which is what BatchNLPPipeline.process_batch() expects. Returning the
        # dict instead breaks collation once "entities" lists differ in length
        # across items in a batch.
        return self.cleaner.clean(raw_tweet)["cleaned_text"]    