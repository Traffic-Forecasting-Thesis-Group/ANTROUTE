"""
Draw a stratified sample of tweets into a CSV for hand-labelling (the gold set).

    python scripts/sample_tweets_for_labeling.py                    # 400 rows, default
    python scripts/sample_tweets_for_labeling.py --n 1000
    python scripts/sample_tweets_for_labeling.py --n 400 --labelers Jaushi,Partner

Writes data/labels/tweet_gold_TEMPLATE.csv (and one _<name>.csv per labeler when
--labelers is given), plus a 200-row overlap block shared by all labelers so
inter-annotator agreement (Cohen's kappa) can be reported.

Why stratified and not random: 43% of the corpus is not Metro Manila at all (it was
retrieved by keyword and caught Bangalore/Hyderabad/LA traffic accounts), and the
Metro Manila half is dominated by two accounts posting in one fixed format. A uniform
sample would spend most of a labeler's time on near-duplicate @MakatiTraffic posts and
on obvious foreign negatives, and would leave almost no examples of the rare classes
(flood, closure, rally) that the classifier most needs. The strata below instead buy
coverage: every stratum is capped, so each contributes examples without any one of
them swamping the sheet.

The sample is reproducible for a given --seed and corpus, so a lost sheet can be
regenerated and a labeler can be re-run against exactly the same rows.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.text.tweet_corpus import corpus_summary, load_tweets, to_frame  # noqa: E402

DEFAULT_RAW_TWITTER = REPO_ROOT / "data" / "raw" / "twitter"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "labels"

# Empty columns the human fills in. Kept short: every extra column is a decision the
# labeler has to make 400 times.
LABEL_COLUMNS = [
    "label_relevant",   # 1 = a Metro Manila road condition/incident, 0 = not
    "label_type",       # crash | stalled | flood | roadwork | closure | rally | heavy_traffic | other
    "label_location",   # the location phrase as written in the tweet, copied verbatim
    "label_lanes",      # 1 | 2 | 3 | blank if not stated
    "label_notes",      # anything ambiguous, for adjudication
]

# Rare classes a uniform sample would almost never surface. Matching here only decides
# whether a row is worth showing a labeler -- the human still assigns the real label.
RARE_CLASS_HINTS = {
    "flood": r"flood|baha|binaha|knee-deep|gutter-deep|not passable",
    "closure": r"closed|closure|sarado|re-?route|rerouting|no entry",
    "rally": r"rally|protest|motorcade|march|kilos|procession",
    "roadwork": r"road ?work|reblocking|excavation|construction|maintenance",
    "stalled": r"stalled|stall |nasiraan|breakdown|overheat",
    "crash": r"crash|collision|aksidente|accident|banggaan|sideswipe|hit and run",
}


def build_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Tag each tweet with the stratum it is sampled from (first match wins)."""
    stratum = pd.Series("other", index=frame.index, dtype=object)
    text = frame["text"]

    # Broadest first, then overwrite with the more specific strata, so the
    # narrow/rare buckets win where a tweet qualifies for several.
    stratum[~frame["metro_manila_hint"]] = "foreign_or_offtopic"
    stratum[frame["metro_manila_hint"]] = "mm_other"
    stratum[frame["metro_manila_hint"] & frame["primary_account"]] = "mm_primary_account"
    stratum[frame["metro_manila_hint"] & (frame["lang"] == "tl")] = "mm_tagalog"

    for name, pattern in RARE_CLASS_HINTS.items():
        hit = frame["metro_manila_hint"] & text.str.contains(pattern, case=False, regex=True, na=False)
        stratum[hit] = f"mm_{name}"

    out = frame.copy()
    out["stratum"] = stratum
    return out


# Share of the sheet each stratum gets. Weighted towards the rare classes and towards
# Tagalog, which the classifier will otherwise never see enough of; foreign negatives
# get a real but small share because the classifier must still learn to reject them.
STRATUM_SHARE = {
    "mm_crash": 0.18,
    "mm_flood": 0.10,
    "mm_closure": 0.10,
    "mm_stalled": 0.10,
    "mm_roadwork": 0.08,
    "mm_rally": 0.06,
    "mm_tagalog": 0.12,
    "mm_primary_account": 0.10,
    "mm_other": 0.10,
    "foreign_or_offtopic": 0.06,
}


def stratified_sample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """`n` rows spread over the strata, topped up from the remainder if a stratum is thin."""
    picked = []
    for stratum, share in STRATUM_SHARE.items():
        pool = frame[frame["stratum"] == stratum]
        if pool.empty:
            continue
        take = min(len(pool), max(1, round(n * share)))
        picked.append(pool.sample(n=take, random_state=seed))

    sample = pd.concat(picked) if picked else frame.head(0)

    # A thin stratum leaves the sheet short of n; fill from whatever is left so the
    # labeler always gets the number of rows they were promised.
    if len(sample) < n:
        remainder = frame.drop(index=sample.index)
        if len(remainder):
            sample = pd.concat([sample, remainder.sample(n=min(len(remainder), n - len(sample)), random_state=seed)])

    return sample.sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)


def to_sheet(sample: pd.DataFrame) -> pd.DataFrame:
    """The sample as the CSV a human actually fills in."""
    sheet = sample[["tweet_id", "created_at", "author", "lang", "stratum", "text", "url"]].copy()
    for column in LABEL_COLUMNS:
        sheet[column] = ""
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-twitter", type=Path, default=DEFAULT_RAW_TWITTER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=400, help="rows per labeler (default 400)")
    parser.add_argument("--overlap", type=int, default=200,
                        help="rows every labeler gets, for inter-annotator agreement")
    parser.add_argument("--labelers", type=str, default="",
                        help="comma-separated names; one sheet each plus the shared overlap block")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading corpus from {args.raw_twitter} ...")
    frame = to_frame(load_tweets(args.raw_twitter))
    if frame.empty:
        raise SystemExit(f"No tweets found under {args.raw_twitter}")

    summary = corpus_summary(frame)
    print(f"  {summary['n_tweets']} tweets, {summary['n_authors']} authors, "
          f"{summary['date_min']} .. {summary['date_max']}")
    print(f"  {summary['n_metro_manila_hint']} mention a Metro Manila place "
          f"({summary['n_metro_manila_hint'] / summary['n_tweets']:.1%}), "
          f"{summary['n_tagalog']} tagged Tagalog")

    frame = build_strata(frame)
    print("\nStratum sizes in the corpus:")
    for stratum, count in frame["stratum"].value_counts().items():
        print(f"  {stratum:22s} {count:6d}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labelers = [name.strip() for name in args.labelers.split(",") if name.strip()]

    if not labelers:
        sheet = to_sheet(stratified_sample(frame, args.n, args.seed))
        path = args.out_dir / "tweet_gold_TEMPLATE.csv"
        sheet.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nWrote {len(sheet)} rows -> {path}")
        return

    # Shared overlap block first, then a disjoint private block per labeler, so the
    # kappa is computed on genuinely common rows and the rest of the budget is not
    # spent labelling the same tweet twice.
    overlap = stratified_sample(frame, args.overlap, args.seed)
    rest = frame[~frame["tweet_id"].isin(overlap["tweet_id"])]

    per_labeler = max(0, args.n - len(overlap))
    for i, name in enumerate(labelers):
        private = stratified_sample(rest, per_labeler, args.seed + 1 + i)
        rest = rest[~rest["tweet_id"].isin(private["tweet_id"])]

        block = pd.concat([overlap, private])
        sheet = to_sheet(block.sample(frac=1.0, random_state=args.seed).reset_index(drop=True))
        sheet.insert(0, "labeler", name)

        path = args.out_dir / f"tweet_gold_{name}.csv"
        sheet.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\nWrote {len(sheet)} rows ({len(overlap)} shared + {len(private)} private) -> {path}")


if __name__ == "__main__":
    main()
