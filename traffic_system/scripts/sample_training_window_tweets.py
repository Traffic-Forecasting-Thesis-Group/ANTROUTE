"""
The tweets that must be hand-placed before the text branch can carry location.

    python scripts/sample_training_window_tweets.py                 # default +/-120 min window
    python scripts/sample_training_window_tweets.py --buffer 60

Writes data/labels/tweet_training_window.csv, in the same shape as the gold-set sheets, so the
same labelling tool opens it.

Why this is a different population from scripts/sample_tweets_for_labeling.py: that one samples
the whole year to score the event-layer classifier. This one is for retraining the model, and a
tweet can only teach the model where there is visual ground truth to learn against -- inside one
of the 20 labelled CCTV sessions. Of the 800 rows in the gold-set sheets, 8 fall inside a
session, so labelling those does nothing for the retrain.

Only tweets that need a *human* are included. A location is left out when it can already be
placed by machine:

  - an MMDA ALERT whose landmark is in configs/event_landmarks.csv
  - a @MakatiTraffic "TRAFFIC UPDATE ... along <road> from <A> to <B>" post, a fixed format
  - an MMDA ALERT whose landmark is merely missing from the gazetteer, which
    scripts/expand_event_gazetteer.py geocodes without anyone reading it

What remains is unstructured prose -- news items, press releases, personal posts -- where a
human has to decide whether a road condition is being reported at all, and where.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.data.alignment import SESSION_STARTS, SESSION_STEPS, VISUAL_SPLIT  # noqa: E402
from src.routing.event_layer import _LOCATION, load_landmarks, match_landmark  # noqa: E402
from src.text.tweet_corpus import load_tweets  # noqa: E402

DEFAULT_RAW_TWITTER = REPO_ROOT / "data" / "raw" / "twitter"
DEFAULT_OUT = REPO_ROOT / "data" / "labels" / "tweet_training_window.csv"
LANDMARKS_CSV = REPO_ROOT / "configs" / "event_landmarks.csv"

LABEL_COLUMNS = ["label_relevant", "label_type", "label_location", "label_lanes", "label_notes"]

# Anything a congestion model could use, not only incidents: a report that a road is slow is
# as much a signal as a report that it is blocked.
TRAFFIC_WORDS = re.compile(
    r"\b(traffic|congest|heavy|slow|stalled|crash|accident|collision|flood|baha|closed|closure|"
    r"re-?rout|lane|stuck|aksidente|sunog|fire|roadwork|reblocking|mabagal)\b",
    re.I,
)
MAKATI_UPDATE = re.compile(r"along (.+?) from", re.I)


def session_windows(buffer_minutes: int) -> List[Tuple[pd.Timestamp, pd.Timestamp, str, str]]:
    """
    Each labelled CCTV session, widened by `buffer_minutes` at both ends.

    The buffer exists because an incident reported shortly before a session still shapes the
    traffic the cameras record once it starts, and one reported shortly after was usually
    building during it.
    """
    windows = []
    for (day, session), split in sorted(VISUAL_SPLIT.items()):
        start = pd.Timestamp.combine(day, SESSION_STARTS[session])
        windows.append((
            start - pd.Timedelta(minutes=buffer_minutes),
            start + pd.Timedelta(minutes=SESSION_STEPS + buffer_minutes),
            split,
            f"{day} {session}",
        ))
    return windows


def needs_a_human(text: str, landmarks: dict) -> bool:
    """True when no rule can place this tweet, so only a reader can."""
    found = _LOCATION.search(text)
    if found and match_landmark(found.group(1).strip(), landmarks):
        return False        # already in the gazetteer
    if found:
        return False        # MMDA ALERT phrase: expand_event_gazetteer.py geocodes it
    if text.startswith("TRAFFIC UPDATE") and MAKATI_UPDATE.search(text):
        return False        # @MakatiTraffic names the road in a fixed format
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-twitter", type=Path, default=DEFAULT_RAW_TWITTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--buffer", type=int, default=120,
                        help="minutes of slack around each 2-hour session (default 120)")
    parser.add_argument("--labeler", type=str, default="", help="name to stamp on the sheet")
    args = parser.parse_args()

    landmarks = load_landmarks(LANDMARKS_CSV)
    windows = session_windows(args.buffer)
    print(f"{len(windows)} labelled CCTV sessions, widened by +/-{args.buffer} min")

    rows = []
    in_window = metro = traffic = machine_placed = 0
    for tweet in load_tweets(args.raw_twitter):
        if tweet.created_at is None:
            continue
        hit = next((w for w in windows if w[0] <= tweet.created_at < w[1]), None)
        if hit is None:
            continue
        in_window += 1
        if not tweet.looks_metro_manila:
            continue
        metro += 1
        if not TRAFFIC_WORDS.search(tweet.text):
            continue
        traffic += 1
        if not needs_a_human(tweet.text, landmarks):
            machine_placed += 1
            continue
        _, _, split, session = hit
        rows.append({
            "tweet_id": tweet.tweet_id,
            "created_at": tweet.created_at,
            "author": tweet.author,
            "lang": tweet.lang,
            "session": session,
            "split": split,
            "text": tweet.text,
            "url": tweet.url,
        })

    print(f"  inside a session window        : {in_window}")
    print(f"  of those, Metro Manila         : {metro}")
    print(f"  of those, traffic-related      : {traffic}")
    print(f"  placed by machine (no labelling): {machine_placed}")
    print(f"  NEEDS A HUMAN                  : {len(rows)}")

    if not rows:
        raise SystemExit("nothing to label")

    sheet = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    for column in LABEL_COLUMNS:
        sheet[column] = ""
    if args.labeler:
        sheet.insert(0, "labeler", args.labeler)

    print("\n  by split: " + ", ".join(f"{k}={v}" for k, v in sheet["split"].value_counts().items()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nwrote {len(sheet)} rows -> {args.out}")


if __name__ == "__main__":
    main()
