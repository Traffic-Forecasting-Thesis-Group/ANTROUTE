"""
Geocode the MMDA incident locations the Event-Aware layer is currently throwing away.

    python scripts/expand_event_gazetteer.py --dry-run        # show what would be added
    python scripts/expand_event_gazetteer.py --top 60         # geocode and write

src/routing/event_layer.py places an incident only when its location text contains a
phrase listed in configs/event_landmarks.csv, and drops the alert otherwise -- a
deliberate choice, since a mis-placed incident would divert a route for no reason. The
cost of that choice is invisible until measured: the gazetteer covers 15 intersections,
so 912 of the 2,280 parsed alerts (40%) never reach routing at all. Nearly all of them
name an EDSA cross-street that simply was never listed (Pioneer, Reliance, Connecticut,
Estrella, White Plains, Kalayaan, ...).

The unmatched locations repeat heavily, so a short list goes a long way: the top 60
landmarks cover 87% of parsed alerts, against 60% today.

Geocoding uses Nominatim at its published 1 request/second limit, the same source as
the existing rows in configs/event_intersections.csv (source=nominatim). Results are
cached so a re-run costs nothing, and anything that geocodes outside Metro Manila is
rejected rather than written -- a wrong coordinate here would move incidents onto roads
they never happened on.
"""

import argparse
import csv
import io
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Landmarks carry Spanish spellings (Muñoz, España) that the Windows console's cp1252
# default mangles or raises on.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.routing.event_layer import _LOCATION, load_landmarks, match_landmark  # noqa: E402
from src.text.tweet_corpus import load_tweets  # noqa: E402

DEFAULT_RAW_TWITTER = REPO_ROOT / "data" / "raw" / "twitter"
LANDMARKS_CSV = REPO_ROOT / "configs" / "event_landmarks.csv"
INTERSECTIONS_CSV = REPO_ROOT / "configs" / "event_intersections.csv"
CACHE_PATH = REPO_ROOT / "data" / "processed" / "geocode_cache.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ANTROUTE-thesis/1.0 (traffic routing research)"
REQUEST_INTERVAL_SECONDS = 1.1  # Nominatim's usage policy is 1 req/s; stay under it

# Metro Manila bounding box. A geocode outside this is a wrong hit (there is an "EDSA"
# in other provinces, and bare cross-street names match all over the country), so it is
# rejected rather than written into the gazetteer.
MM_BOUNDS = (14.35, 14.80, 120.88, 121.18)  # lat_min, lat_max, lon_min, lon_max

_DIRECTION_SUFFIX = re.compile(r"\s+(nb|sb|eb|wb)\b.*$", re.I)
_TRAILING_NOISE = re.compile(r"\s*\((?:[^)]*)\)\s*$")

# Phrases that name a corridor rather than a point on it. An incident reported as just
# "EDSA" cannot be placed: geocoding it yields the road's centroid, which would park
# dozens of incidents on one arbitrary stretch and divert routes that never passed the
# real one. Dropping them is the same conservative choice event_layer.py already makes.
TOO_VAGUE = {"edsa", "c5", "c-5", "commonwealth", "roxas blvd", "taft", "taft ave",
             "slex", "nlex", "skyway", "quezon ave", "espana", "españa", "ortigas ave"}

# Spelling variants of one place. Normalizing these before geocoding keeps "EDSA White
# Plains" and "EDSA Whiteplains" on a single graph intersection instead of two nodes a
# few metres apart, each holding half the incidents.
_ABBREVIATIONS = {
    "ave": "avenue", "aven": "avenue", "st": "street", "blvd": "boulevard",
    "rd": "road", "cor": "corner", "mrt": "", "lrt": "",
}


def canonical_key(landmark: str) -> str:
    """
    A spelling-insensitive key for one physical place.

    Folds accents, expands abbreviations, and drops spaces so 'edsa white plains',
    'edsa whiteplains', 'edsa main ave' and 'edsa main avenue' collapse together.
    """
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", landmark.lower()) if not unicodedata.combining(c)
    )
    words = [_ABBREVIATIONS.get(w, w) for w in re.split(r"[^a-z0-9]+", folded) if w]
    return "".join(w for w in words if w)


def bare_landmark(location_text: str) -> str:
    """'EDSA Pioneer NB' -> 'edsa pioneer'. Direction is handled separately by routing."""
    text = _DIRECTION_SUFFIX.sub("", location_text.strip().lower())
    text = _TRAILING_NOISE.sub("", text)
    return re.sub(r"\s+", " ", text).strip(" ,.-")


def unmatched_landmarks(
    raw_twitter_root: Path, landmarks: Dict[str, str]
) -> Tuple[Counter, int, int, int]:
    """Counted unmatched phrases, plus (n_parsed, n_matched, n_too_vague) for the report."""
    counts: Counter = Counter()
    n_parsed = n_matched = n_vague = 0
    for tweet in load_tweets(raw_twitter_root):
        if "MMDA ALERT" not in tweet.text.upper():
            continue
        found = _LOCATION.search(tweet.text)
        if not found:
            continue
        n_parsed += 1
        location = found.group(1).strip()
        if match_landmark(location, landmarks):
            n_matched += 1
            continue
        bare = bare_landmark(location)
        if not bare:
            continue
        if bare in TOO_VAGUE:
            n_vague += 1
            continue
        counts[bare] += 1
    return counts, n_parsed, n_matched, n_vague


def cluster_spellings(counts: Counter) -> List[Tuple[str, List[str], int]]:
    """
    Group spelling variants of one place, most incidents first.

    Returns (representative, all_spellings, total_count). The representative is the
    most frequent spelling, which is what gets geocoded; every spelling in the cluster
    is then written into the gazetteer pointing at that one intersection.
    """
    clusters: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for landmark, count in counts.items():
        clusters[canonical_key(landmark)].append((landmark, count))

    grouped = []
    for members in clusters.values():
        members.sort(key=lambda pair: -pair[1])
        grouped.append((members[0][0], [name for name, _ in members], sum(c for _, c in members)))
    grouped.sort(key=lambda row: -row[2])
    return grouped


def intersection_label(landmark: str) -> str:
    """'edsa pioneer' -> 'EDSA-Pioneer', matching the existing naming convention."""
    parts = landmark.split()
    if parts and parts[0] == "edsa":
        rest = " ".join(parts[1:]).title()
        return f"EDSA-{rest}" if rest else "EDSA"
    return "-".join(word.title() for word in parts)


def _query_variants(landmark: str) -> List[str]:
    """
    Query forms to try, most specific first.

    An intersection geocodes far more reliably as two named roads than as one phrase,
    so 'edsa pioneer' is asked as 'EDSA, Pioneer Street' before falling back to the
    raw phrase.
    """
    parts = landmark.split()
    variants = []
    if parts and parts[0] == "edsa" and len(parts) > 1:
        cross = " ".join(parts[1:])
        variants.append(f"EDSA, {cross}, Metro Manila, Philippines")
        variants.append(f"{cross}, EDSA, Quezon City, Philippines")
    variants.append(f"{landmark}, Metro Manila, Philippines")
    return variants


def _load_cache() -> Dict[str, Optional[dict]]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: Dict[str, Optional[dict]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def _nominatim(query: str) -> Optional[dict]:
    url = f"{NOMINATIM_URL}?{urllib.parse.urlencode({'q': query, 'format': 'json', 'limit': 1})}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            results = json.load(response)
    except Exception as error:  # network, timeout, rate-limit -- all mean "no result now"
        print(f"    ! {type(error).__name__}: {error}")
        return None
    return results[0] if results else None


def geocode(landmark: str, cache: Dict[str, Optional[dict]]) -> Optional[Tuple[float, float]]:
    """(lat, lon) inside Metro Manila, or None. Cached across runs."""
    if landmark in cache:
        hit = cache[landmark]
        return (hit["lat"], hit["lon"]) if hit else None

    for query in _query_variants(landmark):
        result = _nominatim(query)
        time.sleep(REQUEST_INTERVAL_SECONDS)
        if not result:
            continue
        lat, lon = float(result["lat"]), float(result["lon"])
        lat_min, lat_max, lon_min, lon_max = MM_BOUNDS
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            print(f"    - rejected (outside Metro Manila): {query} -> {lat:.4f},{lon:.4f}")
            continue
        cache[landmark] = {"lat": lat, "lon": lon, "query": query,
                           "display_name": result.get("display_name", "")}
        return lat, lon

    cache[landmark] = None
    return None


def _append_rows(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-twitter", type=Path, default=DEFAULT_RAW_TWITTER)
    parser.add_argument("--top", type=int, default=60, help="how many unmatched landmarks to geocode")
    parser.add_argument("--dry-run", action="store_true", help="report only; geocode nothing, write nothing")
    args = parser.parse_args()

    landmarks = load_landmarks(LANDMARKS_CSV)
    counts, n_parsed, n_matched, n_vague = unmatched_landmarks(args.raw_twitter, landmarks)
    clusters = cluster_spellings(counts)

    print(f"Parsed a location from {n_parsed} MMDA alerts")
    print(f"  matched by the current gazetteer: {n_matched} ({n_matched / n_parsed:.1%})")
    print(f"  unmatched: {n_parsed - n_matched}, of which {n_vague} name only a corridor "
          f"(unplaceable) and {sum(c for _, _, c in clusters)} name a real point")
    print(f"  those resolve to {len(clusters)} distinct places after merging spellings\n")

    targets = clusters[: args.top]
    recoverable = sum(count for _, _, count in targets)
    print(f"Top {len(targets)} places cover {recoverable} of those alerts "
          f"-> would take coverage to {(n_matched + recoverable) / n_parsed:.0%}\n")

    if args.dry_run:
        for representative, spellings, count in targets:
            variants = f"  [{', '.join(s for s in spellings if s != representative)}]" if len(spellings) > 1 else ""
            print(f"  {count:4d}  {representative:34s} -> {intersection_label(representative)}{variants}")
        return

    # Existing intersections, so a landmark that resolves to one already on the graph
    # reuses it instead of creating a duplicate node a few metres away.
    existing_intersections = set()
    if INTERSECTIONS_CSV.exists():
        with INTERSECTIONS_CSV.open(encoding="utf-8", newline="") as f:
            existing_intersections = {row["intersection"] for row in csv.DictReader(f)}

    cache = _load_cache()
    new_landmarks: List[dict] = []
    new_intersections: List[dict] = []
    failed: List[str] = []

    try:
        for i, (representative, spellings, count) in enumerate(targets, start=1):
            label = intersection_label(representative)
            print(f"[{i}/{len(targets)}] {representative} ({count} alerts) -> {label}")
            point = geocode(representative, cache)
            if point is None:
                failed.append(representative)
                print("    ! no usable geocode")
                continue
            lat, lon = point
            print(f"    ok {lat:.6f}, {lon:.6f}")
            # Every spelling of the place points at the one intersection.
            for spelling in spellings:
                new_landmarks.append({"landmark": spelling, "intersection": label})
            if label not in existing_intersections:
                existing_intersections.add(label)
                new_intersections.append({"intersection": label, "lat": f"{lat:.7f}",
                                          "lon": f"{lon:.7f}", "source": "nominatim"})
    finally:
        # Geocoding is slow and rate-limited; never lose it to a late failure.
        _save_cache(cache)

    if new_landmarks:
        _append_rows(LANDMARKS_CSV, ["landmark", "intersection"], new_landmarks)
    if new_intersections:
        _append_rows(INTERSECTIONS_CSV, ["intersection", "lat", "lon", "source"], new_intersections)

    recovered = sum(count for representative, _, count in targets if representative not in failed)
    print(f"\nAdded {len(new_landmarks)} landmarks and {len(new_intersections)} intersections")
    print(f"Failed to geocode {len(failed)}: {', '.join(failed) if failed else '-'}")
    print(f"Event coverage: {n_matched}/{n_parsed} ({n_matched / n_parsed:.0%}) "
          f"-> {n_matched + recovered}/{n_parsed} ({(n_matched + recovered) / n_parsed:.0%})")


if __name__ == "__main__":
    main()
