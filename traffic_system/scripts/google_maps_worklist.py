"""
Turn the Google Maps template into a flat worklist: one row per lookup actually needed.

    python scripts/google_maps_worklist.py --template outputs/appendix/template.csv

The template has three blank duration columns across ~50 rows, which reads as 150 lookups.
It is not:

  * the alternative scenario reuses the recommended trip's origin, destination and window,
    so they share one Google-route timing;
  * where same_route is True both systems chose the same path, so one timing fills both
    columns;
  * a leg ACO could not route, or one the baseline does not define, needs no lookup at all.

This writes the deduplicated list, in an order that groups lookups sharing a departure slot,
with a link that opens the route in Google Maps and the exact cell the answer belongs in. The
point is that whoever does the typing never has to work out which rows are redundant --
getting that wrong is how a sheet ends up with inconsistent timings for the same journey.

Google Maps is the reference. It predicts traffic for a present or future date and time, not
a past one, so the `departure_slot` column is a weekday and time (the window's own May date
has gone). Per lookup, record how the figure was obtained -- scoring refuses rows without it:

    duration_seconds       the travel time read off, in seconds
    google_access_method   e.g. "Google Maps web (desktop), Depart at"
    google_departure_date  the date entered as the departure (YYYY-MM-DD, on the slot's weekday)
    google_queried_on      the date the lookup was made (YYYY-MM-DD)
    google_traffic_model   which figure was read: for the web app's "typically X-Y min" range,
                           e.g. "typical_range_upper"; for the Routes API, its trafficModel.
                           Use one rule for every lookup.

C_optimal is Google's fastest route under its predicted traffic for that slot; that the slot is
a typical weekday rather than the recorded day is a stated limitation of the comparison.
"""

import argparse
import csv
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.routing.route_trials import REFERENCE_FIELDS  # noqa: E402

DURATION_COLUMNS = ("google_duration_seconds", "antroute_google_duration_seconds", "baseline_google_duration_seconds")
FIELDS = ["lookup", "departure_slot", "window", "fill_in", "trip_legs", "maps_url", "start", "stops", "end",
          "duration_seconds", *REFERENCE_FIELDS]
MAX_WAYPOINTS = 9   # what a Google Maps directions link carries on the web


def waypoint_text(value) -> str:
    """The template stores stops as 'lat,lon | lat,lon'."""
    if pd.isna(value) or not str(value).strip():
        return ""
    return " | ".join(p.strip() for p in str(value).split("|") if p.strip())


def maps_url(start: str, stops: str, end: str) -> str:
    """
    A Google Maps directions link (Maps URLs, api=1) with the start, every stop and the end
    pre-filled, for driving. It opens with the whole route plotted; the departure date and time
    are not settable through the link, so they are set once per slot by hand (rows are grouped
    by departure_slot), and the duration read off.
    """
    waypoints = [p.strip() for p in waypoint_text(stops).split("|") if p.strip()]
    if len(waypoints) > MAX_WAYPOINTS:
        raise ValueError(f"{len(waypoints)} waypoints; a Google Maps link takes at most {MAX_WAYPOINTS}")
    url = (f"https://www.google.com/maps/dir/?api=1&origin={quote(start.strip())}"
           f"&destination={quote(end.strip())}&travelmode=driving")
    if waypoints:
        url += "&waypoints=" + quote("|".join(waypoints))
    return url


def apply_worklist(template: pd.DataFrame, worklist_path: Path, template_path: Path) -> None:
    """
    Copy the durations, and how each was obtained, from the filled-in worklist back into the
    template.

    Done here rather than by hand because one timing can belong in several cells -- a shared
    Google-route lookup, or a same_route leg feeding both systems' columns. A blank is left
    blank: a lookup nobody did must not become a zero, which would score as an instant journey.
    A template row fed by lookups that recorded different provenance gets all of them, joined
    with " ; ", and is reported, so mixed methods are visible rather than overwritten.
    """
    if not worklist_path.exists():
        raise SystemExit(f"{worklist_path} does not exist -- generate it first, without --apply")
    work = pd.read_csv(worklist_path, dtype=str, keep_default_na=False)

    for column in DURATION_COLUMNS:
        template[column] = pd.to_numeric(template[column], errors="coerce")
    for field in REFERENCE_FIELDS:
        template[field] = template[field].fillna("").astype(str) if field in template else ""

    index = {(str(r.trip_id), int(r.leg)): i for i, r in enumerate(template.itertuples())}
    filled, blank, unmatched, undocumented, mixed = 0, 0, [], 0, set()
    for r in work.itertuples():
        seconds = pd.to_numeric(getattr(r, "duration_seconds", None), errors="coerce")
        if pd.isna(seconds):
            blank += 1
            continue
        provenance = {f: str(getattr(r, f, "")).strip() for f in REFERENCE_FIELDS}
        if not all(provenance.values()):
            undocumented += 1
        columns = [c.strip() for c in str(r.fill_in).replace("AND", ",").split(",")
                   if c.strip().split(" ")[0].endswith("seconds")]
        columns = [c.split(" ")[0] for c in columns]
        for leg_label in str(r.trip_legs).split(","):
            trip, _, leg = leg_label.strip().partition("/leg")
            position = index.get((trip, int(leg))) if leg.isdigit() else None
            if position is None:
                unmatched.append(leg_label.strip())
                continue
            for column in columns:
                template.iat[position, template.columns.get_loc(column)] = float(seconds)
                filled += 1
            for field, value in provenance.items():
                if not value:
                    continue
                current = template.iat[position, template.columns.get_loc(field)]
                values = [v for v in current.split(" ; ") if v]
                if value not in values:
                    if values:
                        mixed.add(f"{trip}/leg{leg}")
                    template.iat[position, template.columns.get_loc(field)] = " ; ".join(values + [value])

    template.to_csv(template_path, index=False)
    print(f"wrote {filled} timing(s) into {template_path}")
    if blank:
        print(f"  {blank} worklist row(s) still have no duration_seconds -- those cells are left empty")
    if undocumented:
        print(f"  WARNING: {undocumented} timed lookup(s) do not record {', '.join(REFERENCE_FIELDS)}; "
              "evaluate_routing.py skips the trips they feed until they do")
    if mixed:
        print(f"  NOTE: {len(mixed)} leg(s) combine lookups with different provenance: {', '.join(sorted(mixed)[:8])}")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} worklist row(s) name a leg not in the template: "
              f"{', '.join(sorted(set(unmatched))[:8])}")
    for column in DURATION_COLUMNS:
        print(f"  {column:34s} {int(template[column].notna().sum()):3d} of {len(template)} filled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", type=Path, default=REPO_ROOT / "outputs/appendix/template.csv")
    parser.add_argument("--out", type=Path, help="default: lookups.csv beside the template")
    parser.add_argument("--apply", action="store_true",
                        help="read the filled-in worklist back and write the times into the template")
    args = parser.parse_args()

    t = pd.read_csv(args.template)
    out_path = args.out or args.template.with_name("lookups.csv")

    if args.apply:
        apply_worklist(t, out_path, args.template)
        return

    blank = {field: "" for field in REFERENCE_FIELDS}
    rows = []

    # 1. Google's own fastest route, once per distinct journey. Several template rows can
    #    share it (recommended and alternative are the same journey), and they must all get
    #    the same number -- so it is listed once, with every cell it fills.
    key = t["window"].astype(str) + "|" + t["origin"].astype(str) + "|" + t["destination"].astype(str)
    for _, group in t.assign(_key=key).groupby("_key", sort=False):
        first = group.iloc[0]
        rows.append({
            "lookup": "google_own_route",
            "departure_slot": first["google_departure_slot"],
            "window": first["window"],
            "fill_in": "google_duration_seconds",
            "trip_legs": ", ".join(f"{r.trip_id}/leg{r.leg}" for r in group.itertuples()),
            "maps_url": maps_url(first["origin_latlon"], "", first["destination_latlon"]),
            "start": first["origin_latlon"],
            "stops": "",
            "end": first["destination_latlon"],
            "duration_seconds": "",
            **blank,
        })

    # 2. Each system's own path, forced through its waypoints. Where same_route is True the
    #    two paths are identical, so one lookup fills both columns and is listed once.
    for r in t.itertuples():
        same = r.same_route is True or str(r.same_route).strip().lower() == "true"
        has_antroute = not pd.isna(r.antroute_route)
        has_baseline = not pd.isna(r.baseline_route)

        if has_antroute:
            fill = "antroute_google_duration_seconds"
            if same and has_baseline:
                fill += " AND baseline_google_duration_seconds (same path)"
            rows.append({
                "lookup": "antroute_path",
                "departure_slot": r.google_departure_slot,
                "window": r.window,
                "fill_in": fill,
                "trip_legs": f"{r.trip_id}/leg{r.leg}",
                "maps_url": maps_url(r.origin_latlon, r.antroute_waypoints, r.destination_latlon),
                "start": r.origin_latlon,
                "stops": waypoint_text(r.antroute_waypoints),
                "end": r.destination_latlon,
                "duration_seconds": "",
                **blank,
            })
        if has_baseline and not same:
            rows.append({
                "lookup": "baseline_path",
                "departure_slot": r.google_departure_slot,
                "window": r.window,
                "fill_in": "baseline_google_duration_seconds",
                "trip_legs": f"{r.trip_id}/leg{r.leg}",
                "maps_url": maps_url(r.origin_latlon, r.baseline_waypoints, r.destination_latlon),
                "start": r.origin_latlon,
                "stops": waypoint_text(r.baseline_waypoints),
                "end": r.destination_latlon,
                "duration_seconds": "",
                **blank,
            })

    # Grouped by departure slot, so the departure date and time are set once per slot.
    rows.sort(key=lambda d: (str(d["departure_slot"]), str(d["window"]), d["lookup"], d["trip_legs"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_kind = pd.Series([r["lookup"] for r in rows]).value_counts()
    print(f"{len(rows)} lookups -> {out_path}")
    for kind, n in by_kind.items():
        print(f"  {n:3d}  {kind}")
    slots = pd.Series([str(r["departure_slot"]) for r in rows]).nunique()
    print(f"\ngrouped into {slots} departure slots; set the departure once per slot")
    print(f"\nFill duration_seconds and {', '.join(REFERENCE_FIELDS)}, then:")
    print(f"  python scripts/google_maps_worklist.py --template {args.template} --apply")


if __name__ == "__main__":
    main()
