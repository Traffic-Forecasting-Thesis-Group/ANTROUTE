"""
Loading and normalizing the raw X/Twitter corpus under data/raw/twitter.

Every downstream use of the unstructured modality -- the labelling sample, the LLM
pre-labeller, the DistilBERT relevance/event classifier, the geocoder, and the
per-camera text features -- reads the corpus through this module, so they all agree
on what one "tweet" is and on the order tweets are visited in.

Two things the raw feed makes easy to get wrong, and that this module handles once:

1. The corpus is not all Metro Manila. It was retrieved by keyword, so it also contains
   foreign traffic accounts (Bangalore's @blrcitytraffic, Hyderabad's @HYDTP,
   @LAPDCentral, @DEStatePolice, ...) -- roughly 43% of the 46,098 tweets mention no
   Metro Manila place at all. `looks_metro_manila` is a cheap keyword prefilter for
   that; it is a sampling aid and a feature, NOT the final answer, which is what the
   trained relevance classifier is for.

2. `createdAt` is UTC in X's own format ("Sat May 31 16:00:44 +0000 2025"), while every
   timestamp in the rest of ANTROUTE is naive Manila local time (see
   src/data/alignment.to_local). Parsing happens here so no caller has to remember.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pandas as pd

LOCAL_TZ = "Asia/Manila"

# X's created_at format, e.g. "Sat May 31 16:00:44 +0000 2025"
CREATED_AT_FORMAT = "%a %b %d %H:%M:%S %z %Y"

# Cheap "is this even about Metro Manila" prefilter. Deliberately broad: it decides
# which tweets are worth a human's labelling time, so a false positive costs one
# skipped row while a false negative silently removes a real incident from the study.
METRO_MANILA_HINTS = re.compile(
    r"\b(edsa|mmda|metro manila|c-?5|c-?6|slex|nlex|skyway|coastal road|"
    r"quezon ave|quezon city|commonwealth|katipunan|ortigas|shaw|guadalupe|"
    r"magallanes|buendia|ayala|makati|bgc|taguig|pasig|mandaluyong|manila|"
    r"marikina|caloocan|balintawak|cubao|alabang|paranaque|parañaque|pasay|"
    r"roxas blvd|taft|espana|españa|recto|aurora blvd|santolan|kamuning|"
    r"munoz|muñoz|monumento|bicutan|sucat|nagtahan|quiapo|divisoria|valenzuela|"
    r"las pinas|las piñas|muntinlupa|san juan|navotas|malabon|pateros)\b",
    re.I,
)

# Accounts whose posts are overwhelmingly structured incident reports. Used only to
# stratify the labelling sample so the gold set is not swamped by news chatter --
# never as a label in itself.
PRIMARY_TRAFFIC_ACCOUNTS = {
    "mmda",
    "makatitraffic",
    "mmda_trafficnav",
    "dotrph",
    "ltofficialph",
}


@dataclass(frozen=True)
class Tweet:
    """One post, normalized to Manila local time."""

    tweet_id: str
    created_at: Optional[datetime]  # naive Manila local time; None if unparseable
    author: str
    text: str                       # whitespace-collapsed, never empty
    lang: Optional[str]
    url: str
    source_file: str                # for traceability back into data/raw/twitter

    @property
    def looks_metro_manila(self) -> bool:
        return bool(METRO_MANILA_HINTS.search(self.text))

    @property
    def is_primary_traffic_account(self) -> bool:
        return self.author.lower() in PRIMARY_TRAFFIC_ACCOUNTS


def parse_created_at(raw: Optional[str]) -> Optional[datetime]:
    """X's UTC created_at -> naive Manila local time, matching alignment.to_local."""
    if not raw:
        return None
    try:
        aware = datetime.strptime(raw, CREATED_AT_FORMAT)
    except (ValueError, TypeError):
        return None
    return aware.astimezone(tz=pd.Timestamp.now(tz=LOCAL_TZ).tz).replace(tzinfo=None)


def _tweet_records(path: Path) -> Iterator[dict]:
    """The tweet dicts inside one tweets_HHMM_HHMM.json, tolerating both layouts."""
    try:
        with path.open(encoding="utf-8") as f:
            content = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    records = content.get("data", content) if isinstance(content, dict) else content
    if not isinstance(records, list):
        return
    for record in records:
        if isinstance(record, dict):
            yield record


def iter_tweets(raw_twitter_root: Path) -> Iterator[Tweet]:
    """
    Every tweet in the corpus, in sorted file order (date, then time window).

    Sorted rather than filesystem order because scripts/cache_cnn_lstm_features.py
    replays this same walk to line rows up with data/processed/embeddings.pt -- the
    order is load-bearing, not cosmetic.
    """
    seen: set = set()
    for path in sorted(Path(raw_twitter_root).rglob("tweets_*.json")):
        for record in _tweet_records(path):
            text = record.get("text") or record.get("full_text") or ""
            if not text.strip():
                continue
            tweet_id = str(record.get("id") or "")
            # The 2-hour retrieval windows overlap at their edges and a tweet can be
            # returned twice; keeping both would double-count it in every metric.
            if tweet_id and tweet_id in seen:
                continue
            if tweet_id:
                seen.add(tweet_id)
            author = (record.get("author") or {}).get("userName") or ""
            yield Tweet(
                tweet_id=tweet_id,
                created_at=parse_created_at(record.get("createdAt")),
                author=author,
                text=re.sub(r"\s+", " ", text).strip(),
                lang=record.get("lang"),
                url=record.get("url") or "",
                source_file=str(path),
            )


def load_tweets(raw_twitter_root: Path) -> List[Tweet]:
    return list(iter_tweets(raw_twitter_root))


def to_frame(tweets: List[Tweet]) -> pd.DataFrame:
    """Tweets as a DataFrame, with the derived columns the sampler stratifies on."""
    return pd.DataFrame(
        {
            "tweet_id": [t.tweet_id for t in tweets],
            "created_at": [t.created_at for t in tweets],
            "author": [t.author for t in tweets],
            "text": [t.text for t in tweets],
            "lang": [t.lang for t in tweets],
            "url": [t.url for t in tweets],
            "metro_manila_hint": [t.looks_metro_manila for t in tweets],
            "primary_account": [t.is_primary_traffic_account for t in tweets],
        }
    )


def corpus_summary(frame: pd.DataFrame) -> Dict[str, object]:
    """Counts worth reporting in the data chapter."""
    dated = frame["created_at"].dropna()
    return {
        "n_tweets": len(frame),
        "n_metro_manila_hint": int(frame["metro_manila_hint"].sum()),
        "n_primary_account": int(frame["primary_account"].sum()),
        "n_authors": int(frame["author"].nunique()),
        "n_tagalog": int((frame["lang"] == "tl").sum()),
        "date_min": dated.min() if len(dated) else None,
        "date_max": dated.max() if len(dated) else None,
    }
