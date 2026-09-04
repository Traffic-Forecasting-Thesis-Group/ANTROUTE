"""GDELT Historical News Scraper for Metro Manila Traffic Events.

This module provides two ways to collect news for May 2026:
1. GDELT DOC API v2 (Zero credentials required, easy HTTP requests)
2. BigQuery SQL Generator (For querying Google BigQuery public datasets directly)
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import requests

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Search parameters matching our pipeline_context.md requirements
KEYWORDS = '(traffic OR trapik OR accident OR aksidente OR flood OR baha OR "road closure")'
LOCATIONS = '(EDSA OR "Roxas Blvd" OR Manila OR "Quezon City" OR Makati OR Pasay)'
# sourcecountry:RP filters for Philippine sources in GDELT (RP = Philippines country code)
QUERY_STR = f"{KEYWORDS} {LOCATIONS} sourcecountry:RP"


def fetch_gdelt_slice(start_dt: datetime, end_dt: datetime, retries: int = 3, backoff: int = 5) -> list[dict[str, Any]]:
    """Fetch news articles from GDELT DOC API with exponential backoff to handle rate limits (429)."""
    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    end_str = end_dt.strftime("%Y%m%d%H%M%S")

    params = {
        "query": QUERY_STR,
        "mode": "artlist",
        "maxrecords": 250,
        "format": "json",
        "startdatetime": start_str,
        "enddatetime": end_str,
        "sort": "dateasc"
    }

    for attempt in range(1, retries + 1):
        try:
            res = requests.get(GDELT_DOC_API, params=params, timeout=30)
            if res.status_code == 429:
                wait_time = backoff * (2 ** (attempt - 1))
                print(f"Rate limited (429) for window {start_str}. Retrying in {wait_time}s (Attempt {attempt}/{retries})...")
                time.sleep(wait_time)
                continue
            res.raise_for_status()
            data = res.json()
            articles = data.get("articles", [])
            return articles if isinstance(articles, list) else []
        except Exception as e:
            print(f"Warning: GDELT query failed for window {start_str}-{end_str}: {e}")
            time.sleep(backoff)

    return []


def collect_gdelt_may_2026(output_root: Path | None = None) -> list[Path]:
    """Collect news articles day-by-day for May 2026 using GDELT API."""
    if output_root is None:
        output_root = Path(__file__).parent.parent.parent / "data" / "raw" / "news"

    start_date = datetime(2026, 5, 1, 0, 0, 0)
    end_date = datetime(2026, 6, 1, 0, 0, 0)

    output_paths = []
    current = start_date

    print(f"Starting GDELT news retrieval from {start_date} to {end_date}...")

    while current < end_date:
        day_end = current + timedelta(days=1)
        if day_end > end_date:
            day_end = end_date

        print(f"Querying GDELT for {current.strftime('%Y-%m-%d')}...")
        articles = fetch_gdelt_slice(current, day_end)

        records = []
        for idx, art in enumerate(articles):
            title = art.get("title", "")
            url = art.get("url", "")
            seendate = art.get("seendate", current.strftime("%Y%m%d"))

            # Format according to our pipeline's expected news record structure
            records.append({
                "id": f"gdelt_{seendate}_{idx}",
                "created_at": art.get("seendate", current.isoformat()),
                "full_text": f"{title}. Source URL: {url}",
                "title": title,
                "source": art.get("domain", "GDELT"),
                "url": url,
                "language": art.get("language", "English")
            })

        out_dir = output_root / current.strftime("%Y-%m-%d")
        out_dir.mkdir(parents=True, exist_ok=True)

        out_file = out_dir / f"gdelt_{current.strftime('%H%M')}_{day_end.strftime('%H%M')}.json"

        payload = {
            "metadata": {
                "source": "GDELT DOC API v2",
                "window_start": current.isoformat(),
                "window_end": day_end.isoformat(),
                "article_count": len(records)
            },
            "data": records
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(records)} articles to {out_file}")
        output_paths.append(out_file)

        current = day_end
        time.sleep(3)  # Throttle requests to respect GDELT limits

    return output_paths


def get_bigquery_sql() -> str:
    """Returns a ready-to-use Google BigQuery SQL string for querying GDELT public datasets."""
    return """
    -- Query GDELT 2.0 Global Knowledge Graph (GKG) directly in Google BigQuery
    -- Public Dataset: bigquery-public-data.gdelt_bq.gkg_partitioned

    SELECT
        GKGRECORDID AS record_id,
        PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)) AS timestamp,
        DocumentIdentifier AS url,
        SourceCollectionIdentifier AS source,
        Locations AS locations,
        Themes AS themes,
        Organizations AS organizations
    FROM
        `bigquery-public-data.gdelt_bq.gkg_partitioned`
    WHERE
        _PARTITIONDATE BETWEEN '2026-05-01' AND '2026-05-31'
        AND (
            LOWER(Locations) LIKE '%philippines%'
            OR LOWER(Locations) LIKE '%manila%'
            OR LOWER(Locations) LIKE '%quezon city%'
            OR LOWER(Locations) LIKE '%makati%'
            OR LOWER(Locations) LIKE '%edsa%'
        )
        AND (
            LOWER(Themes) LIKE '%traffic%'
            OR LOWER(Themes) LIKE '%transport%'
            OR LOWER(Themes) LIKE '%ACCIDENT%'
            OR LOWER(Themes) LIKE '%FLOOD%'
        )
    LIMIT 1000;
    """


if __name__ == "__main__":
    collect_gdelt_may_2026()
