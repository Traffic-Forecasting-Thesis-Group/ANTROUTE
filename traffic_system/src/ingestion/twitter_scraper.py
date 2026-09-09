"""Collect X posts that align exactly with the CCTV observation windows.

The advanced-search endpoint returns at most 20 posts and its provider advises
against cursor pagination. Each observation window is therefore split into
small, non-overlapping time slices and saved as one auditable JSON dataset.
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytz
import requests
from dotenv import load_dotenv


API_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
MANILA_TZ = pytz.timezone("Asia/Manila")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "twitter"

TRAFFIC_KEYWORDS = (
    'traffic', 'trapik', '"traffic jam"', '"bumper to bumper"', 'standstill',
    '"heavy traffic"', '"mabagal na trapik"', '"mabigat na trapik"',
    '"light traffic"', '"magaan na trapik"', '"moving traffic"', 'dumadaloy',
    'congestion', 'congested', 'gridlock', '"slow moving"', '"creeping traffic"',
    'contraflow', 'counterflow', '"number coding"', '"coding scheme"', 'UVVRP',
    'accident', 'aksidente', 'banggaan', 'collision', 'nabangga', 'nabunggo',
    '"vehicular accident"', '"road crash"', '"road mishap"',
    '"multi-vehicle collision"', '"chain collision"', 'pileup', '"hit and run"',
    'tumakas', 'overturned', 'tumaob', 'natumba', '"motorcycle accident"',
    '"sakay ng motor aksidente"', 'flooding', 'baha', 'bumabaha', '"flash flood"',
    '"biglaang baha"', '"heavy rain"', '"malakas na ulan"', 'storm', 'bagyo',
    'typhoon', '"waist-deep"', 'tuhod', 'baywang', 'impassable', '"hindi madaanan"',
    'landslide', 'guho', '"pagguho ng lupa"', '"PAGASA warning"', '"signal no."',
    '"road closure"', '"sarado ang daan"', '"lane closure"', '"isang lane lang"',
    '"road construction"', '"kalsada gawa"', 'roadwork', '"ongoing repair"',
    'detour', '"alternate route"', '"alternative route"', 'rerouting', 'biniyahe',
    '"underpass closed"', '"flyover closed"', 'pothole', 'guwang', '"butas sa kalsada"',
    'protest', 'rally', 'welga', 'demonstration', 'martsa', '"barangay fiesta"',
    'procession', 'parada', 'motorcade', 'strike', '"transport strike"',
    '"tigil pasada"', 'fire', 'sunog', '"fire truck"', 'bumbero',
    '"emergency response"', 'ambulansya', 'explosion', 'sabog', 'breakdown',
    '"nasira ang sasakyan"', '"stalled vehicle"', '"tumigil sa gitna"',
    '"flat tire"', 'overheating', '"nag-overheat"', 'towed',
)
LOCATION_ANCHORS = (
    'EDSA', '"EDSA Guadalupe"', '"EDSA Ortigas"', '"EDSA Cubao"',
    '"EDSA Kamuning"', '"EDSA Balintawak"', '"EDSA Taft"', '"Roxas Boulevard"',
    '"Roxas Blvd"', 'Makati', 'Pasay', '"Quezon City"', 'Manila', 'Ortigas',
    'Pasig', 'Novaliches', 'Navotas', 'Mandaluyong', 'Paranaque', '"Las Pinas"',
    'Muntinlupa', 'San Juan', 'Taguig', 'Marikina', 'Caloocan', 'Malabon', 'Valenzuela',
    '"Metro Manila"', 'Philippines', 'Pilipinas',
)
MMDA_TERMS = (
    '#MMDAAlert', '#MetroManila', '#TrafficUpdate', '#EDSAUpdate', '#EDSATraffic',
    '#RoadClosure', '#TrafficAdvisory', '"MMDA advisory"', '"traffic advisory"',
    'Metrobase',
)


def _or_group(terms: tuple[str, ...]) -> str:
    return '(' + ' OR '.join(terms) + ')'


INCIDENTS = _or_group(TRAFFIC_KEYWORDS)
LOCATIONS = _or_group(LOCATION_ANCHORS)
MMDA_TAGS = _or_group(MMDA_TERMS)


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
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
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

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
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
        entry: dict[str, Any] = {
            "start": cursor.isoformat(), "end": slice_end.isoformat(),
        }
        for attempt in range(1, 4):
            try:
                response = client.get(
                    API_URL,
                    headers=headers,
                    params={"query": build_query(cursor, slice_end), "queryType": "Latest"},
                    timeout=30,
                )
                entry["status_code"] = response.status_code
                if response.status_code == 429 and attempt < 3:
                    wait_time = 5 * (2 ** (attempt - 1))
                    print(f"Rate limited for {cursor.isoformat()}; retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
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
                if len(tweets) >= 20:
                    entry["warning"] = "Result cap may have been reached; rerun with a smaller slice_minutes value."
                break
            except (requests.RequestException, ValueError) as error:
                entry["error"] = str(error)
                if attempt < 3:
                    time.sleep(5 * (2 ** (attempt - 1)))
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


def collect_full_month() -> list[Path]:
    """Collect tweets from June 2025 through May 2026 in daily batches."""
    outputs = []

    # Use a half-open interval so every instant in the requested period is collected.
    start_date = MANILA_TZ.localize(datetime(2025, 11, 27, 22, 0, 0))
    end_date = MANILA_TZ.localize(datetime(2026, 6, 1, 0, 0, 0))

    current_start = start_date
    batch_duration = timedelta(hours=2)
    while current_start < end_date:
        day_end = min(current_start + timedelta(days=1), end_date)
        batch_start = current_start
        while batch_start < day_end:
            batch_end = min(batch_start + batch_duration, day_end)
            print(f"Collecting {batch_start:%Y-%m-%d %H:%M}–{batch_end:%H:%M} Asia/Manila")
            outputs.append(retrieve_twitter_data(batch_start, batch_end))
            batch_start = batch_end
        current_start = day_end

    return outputs


if __name__ == "__main__":
    collect_full_month()
