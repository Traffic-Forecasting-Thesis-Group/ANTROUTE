"""
Turn the Apple Maps template into a flat worklist: one row per lookup actually needed.

    python scripts/apple_maps_worklist.py --template outputs/appendix/template.csv

The template has three blank columns across 50 rows, which reads as 150 lookups. It is not:

  * the alternative scenario reuses the recommended trip's origin, destination and window,
    so they share one Apple-route timing;
  * where same_route is True both systems chose the same path, so one timing fills both
    columns;
  * a leg ACO could not route, or one the baseline does not define, needs no lookup at all.

This writes the deduplicated list, in an order that groups lookups sharing a departure time,
with the exact text to paste into Apple Maps and the exact cell the answer belongs in. The
point is that whoever does the typing never has to work out which rows are redundant --
getting that wrong is how a sheet ends up with inconsistent timings for the same journey.

Apple Maps predicts typical traffic for a weekday and time, not for a past date, so the
`leave_at` column is a weekday and time (the window's own date is in May and has gone). That
is a stated limitation of the comparison, not an oversight: C_optimal is Apple's fastest route
under typical conditions for that slot.
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

FIELDS = ["lookup", "leave_at", "window", "fill_in", "trip_legs", "maps_url", "start", "stops", "end", "eta_seconds"]


def waypoint_text(value) -> str:
    """The template stores stops as 'lat,lon | lat,lon'; Apple Maps wants them one at a time."""
    if pd.isna(value) or not str(value).strip():
        return ""
    return " | ".join(p.strip() for p in str(value).split("|") if p.strip())


def maps_url(start: str, stops: str, end: str) -> str:
    """
    A clickable maps.apple.com link with the start, every stop, and the end pre-filled.

    Without this, doing the lookup by hand means typing or pasting 2-6 coordinates into the
    app per row, 109 times. Apple's documented link scheme (saddr / daddr, stops chained with
    "+to:", dirflg=d for driving) opens straight into the Maps app with the whole route
    already plotted -- all that is left to do by hand is setting "Leave at" (once per slot,
    not once per row, since rows are grouped by leave_at) and reading off the ETA. It does not
    set the departure time itself; Apple's link scheme has no parameter for that.
    """
    destinations = [p.strip() for p in waypoint_text(stops).split("|") if p.strip()] + [end.strip()]
    return f"https://maps.apple.com/?saddr={start.strip()}&daddr={'+to:'.join(destinations)}&dirflg=d"


def apply_worklist(template: pd.DataFrame, worklist_path: Path, template_path: Path) -> None:
    """
    Copy the timings from the filled-in worklist back into the template's three columns.

    Done here rather than by hand because one timing can belong in several cells -- a shared
    Apple-route lookup, or a same_route leg feeding both systems' columns -- and transcribing
    109 numbers into 150 cells by hand is where the errors would come from. A blank is left
    blank: a lookup nobody did must not become a zero, which would score as an instant journey.
    """
    if not worklist_path.exists():
        raise SystemExit(f"{worklist_path} does not exist -- generate it first, without --apply")
    work = pd.read_csv(worklist_path)

    for column in ("apple_eta_seconds", "antroute_apple_eta_seconds", "baseline_apple_eta_seconds"):
        template[column] = pd.to_numeric(template[column], errors="coerce")

    index = {(str(r.trip_id), int(r.leg)): i for i, r in enumerate(template.itertuples())}
    filled, blank, unmatched = 0, 0, []
    for r in work.itertuples():
        seconds = pd.to_numeric(getattr(r, "eta_seconds", None), errors="coerce")
        if pd.isna(seconds):
            blank += 1
            continue
        columns = [c.strip() for c in str(r.fill_in).replace("AND", ",").split(",")
                   if c.strip().endswith("seconds")]
        for leg_label in str(r.trip_legs).split(","):
            trip, _, leg = leg_label.strip().partition("/leg")
            position = index.get((trip, int(leg))) if leg.isdigit() else None
            if position is None:
                unmatched.append(leg_label.strip())
                continue
            for column in columns:
                template.iat[position, template.columns.get_loc(column)] = float(seconds)
                filled += 1

    template.to_csv(template_path, index=False)
    print(f"wrote {filled} timing(s) into {template_path}")
    if blank:
        print(f"  {blank} worklist row(s) still have no eta_seconds -- those cells are left empty")
    if unmatched:
        print(f"  WARNING: {len(unmatched)} worklist row(s) name a leg not in the template: "
              f"{', '.join(sorted(set(unmatched))[:8])}")
    for column in ("apple_eta_seconds", "antroute_apple_eta_seconds", "baseline_apple_eta_seconds"):
        print(f"  {column:30s} {int(template[column].notna().sum()):3d} of {len(template)} filled")


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

    rows = []

    # 1. Apple's own fastest route, once per distinct journey. Several template rows can
    #    share it (recommended and alternative are the same journey), and they must all get
    #    the same number -- so it is listed once, with every cell it fills.
    key = t["window"].astype(str) + "|" + t["origin"].astype(str) + "|" + t["destination"].astype(str)
    for _, group in t.assign(_key=key).groupby("_key", sort=False):
        first = group.iloc[0]
        rows.append({
            "lookup": "apple_own_route",
            "leave_at": first["apple_leave_at"],
            "window": first["window"],
            "fill_in": "apple_eta_seconds",
            "trip_legs": ", ".join(f"{r.trip_id}/leg{r.leg}" for r in group.itertuples()),
            "maps_url": maps_url(first["origin_latlon"], "", first["destination_latlon"]),
            "start": first["origin_latlon"],
            "stops": "",
            "end": first["destination_latlon"],
            "eta_seconds": "",
        })

    # 2. Each system's own path, forced through its waypoints. Where same_route is True the
    #    two paths are identical, so one lookup fills both columns and is listed once.
    for r in t.itertuples():
        same = r.same_route is True or str(r.same_route).strip().lower() == "true"
        has_antroute = not pd.isna(r.antroute_route)
        has_baseline = not pd.isna(r.baseline_route)

        if has_antroute:
            fill = "antroute_apple_eta_seconds"
            if same and has_baseline:
                fill += " AND baseline_apple_eta_seconds (same path)"
            rows.append({
                "lookup": "antroute_path",
                "leave_at": r.apple_leave_at,
                "window": r.window,
                "fill_in": fill,
                "trip_legs": f"{r.trip_id}/leg{r.leg}",
                "maps_url": maps_url(r.origin_latlon, r.antroute_waypoints, r.destination_latlon),
                "start": r.origin_latlon,
                "stops": waypoint_text(r.antroute_waypoints),
                "end": r.destination_latlon,
                "eta_seconds": "",
            })
        if has_baseline and not same:
            rows.append({
                "lookup": "baseline_path",
                "leave_at": r.apple_leave_at,
                "window": r.window,
                "fill_in": "baseline_apple_eta_seconds",
                "trip_legs": f"{r.trip_id}/leg{r.leg}",
                "maps_url": maps_url(r.origin_latlon, r.baseline_waypoints, r.destination_latlon),
                "start": r.origin_latlon,
                "stops": waypoint_text(r.baseline_waypoints),
                "end": r.destination_latlon,
                "eta_seconds": "",
            })

    # Grouped by departure slot: Apple Maps keeps the "Leave at" setting between searches, so
    # doing one slot at a time saves re-entering it on every lookup.
    rows.sort(key=lambda d: (str(d["leave_at"]), str(d["window"]), d["lookup"], d["trip_legs"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_kind = pd.Series([r["lookup"] for r in rows]).value_counts()
    print(f"{len(rows)} lookups -> {out_path}")
    for kind, n in by_kind.items():
        print(f"  {n:3d}  {kind}")
    slots = pd.Series([str(r["leave_at"]) for r in rows]).nunique()
    print(f"\ngrouped into {slots} departure slots; set 'Leave at' once per slot")
    print("\nFill the eta_seconds column, then:")
    print(f"  python scripts/apple_maps_worklist.py --template {args.template} --apply")


if __name__ == "__main__":
    main()
