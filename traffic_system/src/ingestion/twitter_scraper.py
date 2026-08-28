"""Collect X posts that align exactly with the CCTV observation windows.

The advanced-search endpoint returns at most 20 posts and its provider advises
against cursor pagination. Each observation window is therefore split into
small, non-overlapping time slices and saved as one auditable JSON dataset.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz
import requests
from dotenv import load_dotenv


API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
MANILA_TZ = pytz.timezone("Asia/Manila")
WINDOW_DATES = (
    (2026, 5, 4), (2026, 5, 6), (2026, 5, 8),
    (2026, 5, 11), (2026, 5, 13), (2026, 5, 15),
    (2026, 5, 18), (2026, 5, 20), (2026, 5, 22), (2026, 5, 25),
)
PEAK_HOURS = ((7, 9), (17, 19))

INCIDENTS = (
    '(traffic OR trapik OR "traffic jam" OR "bumper to bumper" OR standstill OR '
    '"heavy traffic" OR "mabagal na trapik" OR "mabigat na trapik" OR '
    '"light traffic" OR "magaan na trapik" OR "moving traffic" OR dumadaloy OR '
    'congestion OR congested OR gridlock OR "slow moving" OR "creeping traffic" OR '
    'contraflow OR counterflow OR "number coding" OR "coding scheme" OR UVVRP OR '
    'accident OR aksidente OR banggaan OR collision OR nabangga OR nabunggo OR '
    '"vehicular accident" OR "road crash" OR "road mishap" OR pileup OR '
    'flooding OR baha OR bumabaha OR "flash flood" OR "heavy rain" OR "malakas na ulan" OR '
    'storm OR bagyo OR typhoon OR impassable OR "hindi madaanan" OR landslide OR guho OR '
    '"road closure" OR "sarado ang daan" OR "lane closure" OR "road construction" OR '
    'roadwork OR detour OR rerouting OR pothole OR protest OR rally OR strike OR '
    '"transport strike" OR fire OR sunog OR ambulansya OR explosion OR breakdown OR '
    '"stalled vehicle" OR "flat tire" OR overheating OR towed)'
)
LOCATIONS = '(EDSA OR "Roxas Boulevard" OR "Roxas Blvd" OR Makati OR Pasay OR "Quezon City" OR Manila)'
MMDA_TAGS = '(#MMDAAlert OR #MetroManila OR #TrafficUpdate OR #EDSAUpdate OR #EDSATraffic OR #RoadClosure OR #TrafficAdvisory OR "MMDA advisory" OR "traffic advisory" OR Metrobase)'


def build_query(start_dt: datetime, end_dt: datetime) -> str:
    """Build a time-bounded traffic search query using Unix timestamps (UTC)."""
    return (
        f"(({INCIDENTS} {LOCATIONS}) OR {MMDA_TAGS}) "
        f"since_time:{int(start_dt.timestamp())} until_time:{int(end_dt.timestamp())} "
        "-is:retweet -is:reply"
    )


def _parse_created_at(tweet: dict[str, Any]) -> datetime | None:
    """Parse provider timestamps defensively; unknown formats are retained."""
    value = tweet.get("createdAt") or tweet.get("created_at")
    if not value:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=pytz.UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_in_window(tweet: dict[str, Any], start_dt: datetime, end_dt: datetime) -> bool:
    created_at = _parse_created_at(tweet)
    if created_at is None:
        return True  # Metadata makes timestamps that could not be parsed auditable.
    if created_at.tzinfo is None:
        created_at = pytz.UTC.localize(created_at)
    return start_dt <= created_at.astimezone(pytz.UTC) < end_dt


def retrieve_twitter_data(
    start_dt: datetime,
    end_dt: datetime,
    *,
    slice_minutes: int = 5,
    output_root: Path | str = Path("data/raw/twitter"),
    session: requests.Session | None = None,
) -> Path:
    """Retrieve and organize posts for one CCTV interval.

    The supplied timestamps must be timezone-aware. Five-minute slices avoid
    unsupported cursor pagination and make every request auditable.
    """
    if start_dt.tzinfo is None or end_dt.tzinfo is None:
        raise ValueError("start_dt and end_dt must be timezone-aware")
    if end_dt <= start_dt or slice_minutes <= 0:
        raise ValueError("end_dt must be later than start_dt and slice_minutes must be positive")

    load_dotenv()
    api_key = os.getenv("TWITTER_API_KEY")
    if not api_key:
        raise RuntimeError("TWITTER_API_KEY is not set. Add it to traffic_system/.env.")

    client = session or requests.Session()
    headers = {"X-API-Key": api_key}
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    slice_log: list[dict[str, Any]] = []
    cursor = start_dt

    while cursor < end_dt:
        slice_end = min(cursor + timedelta(minutes=slice_minutes), end_dt)
        response = client.get(
            API_URL,
            headers=headers,
            params={"query": build_query(cursor, slice_end), "queryType": "Latest"},
            timeout=30,
        )
        entry: dict[str, Any] = {
            "start": cursor.isoformat(), "end": slice_end.isoformat(),
            "status_code": response.status_code,
        }
        try:
            response.raise_for_status()
            payload = response.json()
            tweets = payload.get("tweets", []) if isinstance(payload, dict) else []
            accepted = 0
            for tweet in tweets:
                if not isinstance(tweet, dict) or not _is_in_window(tweet, cursor, slice_end):
                    continue
                tweet_id = str(tweet.get("id") or "")
                unique_key = tweet_id or json.dumps(tweet, sort_keys=True, ensure_ascii=False)
                if unique_key not in seen_ids:
                    records.append(tweet)
                    seen_ids.add(unique_key)
                    accepted += 1
            entry.update({"returned": len(tweets), "accepted": accepted})
            # The provider documents advanced-search pagination as unreliable.
            # A response is only potentially truncated when it reaches its
            # documented 20-post result limit, not merely when it has a cursor.
            if len(tweets) >= 20:
                entry["warning"] = "Result cap may have been reached; rerun with a smaller slice_minutes value."
        except (requests.RequestException, ValueError) as error:
            entry["error"] = str(error)
        slice_log.append(entry)
        cursor = slice_end

    output_dir = Path(output_root) / start_dt.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"tweets_{start_dt.strftime('%H%M')}_{end_dt.strftime('%H%M')}.json"
    metadata = {
        "source": "twitterapi.io advanced_search", "timezone": "Asia/Manila",
        "window_start": start_dt.isoformat(), "window_end": end_dt.isoformat(),
        "slice_minutes": slice_minutes, "tweet_count": len(records), "slices": slice_log,
    }
    output_path.write_text(json.dumps({"metadata": metadata, "data": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} posts to {output_path}")
    return output_path


def collect_cctv_matched_windows() -> list[Path]:
    """Collect the 20 specified M/W/F morning and evening CCTV windows."""
    outputs = []
    for year, month, day in WINDOW_DATES:
        for start_hour, end_hour in PEAK_HOURS:
            start = MANILA_TZ.localize(datetime(year, month, day, start_hour))
            end = MANILA_TZ.localize(datetime(year, month, day, end_hour))
            print(f"Collecting {start:%Y-%m-%d %H:%M}–{end:%H:%M} Asia/Manila")
            outputs.append(retrieve_twitter_data(start, end))
    return outputs


if __name__ == "__main__":
    collect_cctv_matched_windows()
