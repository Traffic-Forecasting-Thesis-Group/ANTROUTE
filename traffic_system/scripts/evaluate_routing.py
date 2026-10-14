"""
Evaluate ANTROUTE against the baseline with Google Maps as the reference, as the thesis specifies
(Sections 3.8-3.9, Appendices 1-3). Scope: Metro Manila.

The baseline is Improved ACO (Cheng 2023, doi:10.1155/2023/7651100), the algorithm the thesis
names, implemented as published in src/routing/baseline_iaco.py and adapted to this graph in
src/routing/baseline_router.py. See the Baseline class below for what the paper defines per
scenario and what is an adaptation.

Each system plans on its own signal -- ANTROUTE on predicted risk plus the Event-Aware layer,
the baseline on observed camera congestion -- and both are then timed by Google Maps, which
neither plans on, so neither system is optimising the measure it is judged by.

    # 1. route the test trips with both systems and write the Google Maps lookup sheet
    python scripts/evaluate_routing.py --risk-edges data/processed/risk_scores/risk_edges.csv \\
        --auto-labels ... --template google_maps.csv

    # 2. fill in the google_* columns (see below), then score
    python scripts/evaluate_routing.py --risk-edges data/processed/risk_scores/risk_edges.csv \\
        --google-maps google_maps.csv

Trips (--n-trials per scenario, default 10 as in Appendices 1-3) are drawn from the risk file's
held-out --split (default "test", per Section 3.3) and from CCTV intersections that are
connected on the road graph; scripts/citywide_trips.py adds trips anywhere in Metro Manila,
routed on the whole-city graph:

    recommended        best ACO route, origin -> destination
    alternative        next-best ACO route for the same origin/destination/window
    multi_destination  --n-stops CCTV intersections visited in order, routed leg by leg
    incident_exposed   an incident in force mid-route (see src/routing/route_trials.py)

The sheet has one row per leg (single-destination trips have one). Per row, in Google Maps
(Driving, departing at google_departure_slot -- a weekday and time; Google only predicts
traffic for a present or future date, so the window's own May date cannot be used):

    google_duration_seconds            origin -> destination, Google's own fastest route
    antroute_google_duration_seconds   origin -> antroute_waypoints (in order) -> destination
    baseline_google_duration_seconds   origin -> baseline_waypoints (in order) -> destination

and how each was obtained, which scoring requires and metrics.json reports (REFERENCE_FIELDS):

    google_access_method    e.g. "Google Maps web (desktop), Depart at" or
                            "Routes API computeRoutes, TRAFFIC_AWARE_OPTIMAL"
    google_departure_date   the actual date entered as the departure (YYYY-MM-DD), a date
                            falling on the slot's weekday
    google_queried_on       the date the lookup was made (YYYY-MM-DD)
    google_traffic_model    which figure was read, e.g. "best_guess", or for the web app's
                            "typically X-Y min" range "typical_range_upper" -- one rule for
                            every row

When same_route is True one of the two system columns is enough. An alternative trip's
google_duration_seconds may be left blank: it is the same lookup as the recommended trip's.

Scoring (stored routes are re-scored as-is, ACO is not re-run):

    Route Optimality     = sum(google_duration_seconds) / sum(<system>_google_duration_seconds) * 100
    congestion exposure  = sum of predicted Risk(u, v) over the route, under the window's risk
    event exposure       = sum of Event(u, v) over the route, under the incidents in force then
    avoidable event exp. = the same without the edges at the trip's own stops (unavoidable)
    route length         = descriptive only
    significance         Shapiro-Wilk, Wilcoxon signed-rank, paired t-test (if normal),
                         d = baseline - proposed, Δ% (Equations 7-8)

ETA is not a research question or metric: neither system's own travel-time estimate is scored.

Model provenance is required: risk_summary.json (written beside risk_edges.csv by
scripts/predict_congestion_risk.py) names the checkpoint and whether the text branch was used,
and it is copied into metrics.json. Without it the run stops (--allow-unknown-model overrides,
and says so in the report).

A trip the baseline cannot route is reported in the run's summary and left out of the paired
comparison rather than dropped silently, so the baseline's route-found rate stays visible.

Outputs (in --out-dir):
    trials.csv                           one row per (system, trip)
    appendix1_route_optimality.csv       per-trial C_optimal, C_predicted, Route Optimality
    appendix2_exposure.csv               per-trial congestion, event and avoidable event exposure, length
    appendix3_routing_results.csv        per-trial path, length, congestion and event exposure
    appendix3_route_optimality_pairs.csv per-trial paired Route Optimality, d_i, Δ%
    appendix3_significance.csv           per scenario and metric: Shapiro-Wilk, Wilcoxon, t-test
    per_trip.csv                         per trip: both systems on every metric, Δ% and the better one
    metrics.json                         all of the above plus the model, reference and event-layer records
"""

import argparse
import collections
import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime
from heapq import heappop, heappush
from zoneinfo import ZoneInfo
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.data.alignment import WINDOW_STEPS  # noqa: E402
from src.data.graph_data import DEFAULT_K, build_subgraph  # noqa: E402
from src.routing.aco_routing import AntColonyConfig, ant_colony_shortest_path, remaining_cost_to  # noqa: E402
from src.routing.baseline_iaco import IacoConfig, IacoGraph  # noqa: E402
from src.routing.baseline_router import (  # noqa: E402
    IACO,
    BaselineInputs,
    build_inputs,
    camera_flow,
    camera_observations,
    load_vehicle_counts,
    plan_baseline,
)
from src.routing.dynamic_weight import (  # noqa: E402
    DEFAULT_LAMBDA,
    DEFAULT_MISSING_RISK,
    edge_distances,
    load_risk_edges,
    window_key,
    weighted_graph_from_risk_file,
)
from src.routing.event_layer import (  # noqa: E402
    DEFAULT_MU,
    active_at,
    apply_events,
    describe_event_layer,
    event_impact,
    load_landmarks,
    parse_alerts,
)
from src.routing.eta_engine import load_free_flow_seconds  # noqa: E402
from src.routing.metrics import (  # noqa: E402
    DESCRIPTIVE_METRICS,
    TESTED_METRICS,
    compute_metrics_by_scenario,
    route_optimality,
)
from src.routing.route_trials import (  # noqa: E402
    ALTERNATIVE,
    ANTROUTE,
    BASELINE,
    INCIDENT_EXPOSED,
    MULTI_DESTINATION,
    RECOMMENDED,
    REFERENCE_FIELDS,
    SCENARIOS,
    SYSTEMS,
    pick_waypoints,
    score_trip,
)
from src.routing.significance import HIGHER_IS_BETTER, Q3_FAMILY, compare_paired, holm_adjust, relative_difference  # noqa: E402

TRIP_PREFIX = {RECOMMENDED: "R", ALTERNATIVE: "A", MULTI_DESTINATION: "M", INCIDENT_EXPOSED: "E"}
LOCAL_TZ = "Asia/Manila"
MAX_SNAP_KM = 2.0  # how far an incident landmark may sit from this subgraph and still count


# ---------------------------------------------------------------- the event layer


class Events:
    """
    The Event-Aware layer, part of the method, applied to the routes this evaluation scores.

    Without this the appendix would measure risk-aware routing only: ANTROUTE's graph would
    carry the STGNN's predicted congestion but none of the reported incidents, which are the
    "road-event conditions" of RQ2. The app applies the same layer with the same mu
    (app/config.py EVENT_MU, event_layer.DEFAULT_MU); describe_event_layer records the cost
    term, strength and decay in metrics.json.

    Incidents are evaluated as of the routing window itself, the same clock the risk snapshot
    uses, so both signals describe one moment. Only ANTROUTE's graph gets them: the baseline
    is Cheng's method planning on observed traffic, and reported incidents are precisely what
    ANTROUTE adds over it.
    """

    def __init__(self, graph, spatial_dir: Path, mu: float, enabled: bool = True):
        # `enabled` governs whether the penalty reaches the router, NOT whether incidents are
        # known. The ablation has to study the same trips in both arms, and the
        # incident_exposed scenario is selected by where the incidents are -- so with
        # --no-events the feed is still parsed and placed, and only applied() stands down.
        # Folding the two together meant the ablation's control arm sampled different trips
        # from its treatment arm, which is not an ablation.
        self.mu = mu
        self.enabled = enabled
        self.events = []
        self.intersections: Dict[str, int] = {}
        self._impact: Dict[str, Optional[np.ndarray]] = {}
        self._live: Dict[str, list] = {}
        landmarks = load_landmarks(REPO_ROOT / "configs/event_landmarks.csv")
        raw_twitter = REPO_ROOT / "data/raw/twitter"
        if raw_twitter.exists() and landmarks:
            self.events = parse_alerts(raw_twitter, landmarks)
        self.intersections = self._place(graph, spatial_dir)

    @staticmethod
    def _place(graph, spatial_dir: Path) -> Dict[str, int]:
        """
        Incident label -> graph index, for this subgraph.

        Starts from the cameras, whose indices are already known, and adds the non-camera
        landmarks the MMDA feed names, snapped to the nearest node. A landmark further than
        MAX_SNAP_KM from this subgraph is dropped rather than pulled onto it: the subgraph is
        a corridor, and an incident kilometres away does not belong on it.
        """
        placed: Dict[str, int] = dict(graph.camera_nodes)
        path = REPO_ROOT / "configs/event_intersections.csv"
        if not path.exists():
            return placed
        coords = pd.read_csv(spatial_dir / "full_network_static_features.csv",
                             usecols=["node_id", "lat", "lon"]).set_index("node_id").reindex(graph.node_ids)
        lat, lon = np.radians(coords["lat"].to_numpy()), np.radians(coords["lon"].to_numpy())
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                label = row["intersection"].strip()
                if label in placed:
                    continue
                la, lo = np.radians(float(row["lat"])), np.radians(float(row["lon"]))
                h = np.sin((lat - la) / 2) ** 2 + np.cos(la) * np.cos(lat) * np.sin((lon - lo) / 2) ** 2
                km = 2 * 6371.0 * np.arcsin(np.sqrt(h))
                best = int(np.argmin(km))
                if km[best] <= MAX_SNAP_KM:
                    placed[label] = best
        return placed

    def live(self, window: str) -> list:
        """
        (event, strength) pairs in force at this window.

        Reported whether or not the penalty is enabled: in the ablation's control arm an
        incident is still in force, the router simply is not told. Suppressing it here would
        make the two arms look like they met different traffic.
        """
        if not self.events:
            return []
        if window not in self._live:
            try:
                when = datetime.fromisoformat(window)
            except ValueError:
                self._live[window] = []
                return []
            when = when if when.tzinfo else when.replace(tzinfo=ZoneInfo(LOCAL_TZ))
            self._live[window] = active_at(self.events, when)
        return self._live[window]

    def impact(self, graph, window: str) -> Optional[np.ndarray]:
        if window not in self._impact:
            live = self.live(window)
            self._impact[window] = event_impact(graph, live, self.intersections) if live else None
        return self._impact[window]

    def applied(self, wg, graph, window: str):
        """
        ANTROUTE's graph with this window's incidents penalised.

        Unchanged when there are none, and unchanged under --no-events -- that single
        difference is the whole of the ablation, so it is the only place `enabled` is read.
        """
        if not self.enabled:
            return wg
        impact = self.impact(graph, window)
        return wg if impact is None else apply_events(wg, impact, mu=self.mu)

    def nodes_live(self, graph, window: str) -> set:
        """
        Graph indices of the intersections carrying an incident penalty at this window.

        Read off the impact array rather than the event list so it cannot drift from what
        routing is actually penalised by: event_impact drops incidents whose landmark is not
        on this subgraph, and an intersection missing from that array is one no route can be
        diverted away from.
        """
        impact = self.impact(graph, window)
        if impact is None:
            return set()
        src, dst = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
        penalised = impact > 0
        return set(src[penalised].tolist()) | set(dst[penalised].tolist())

    def on_route(self, wg, graph, window: str, legs: Sequence[Sequence[int]]) -> int:
        """
        How many of this route's edges carry an incident penalty AND could have been avoided --
        0 when it avoided every avoidable one.

        An edge touching one of the trip's own stops is left out: an incident at the origin or
        destination penalises the edges there, and no route can avoid its own endpoints.
        Counting them made every route through an incident at its endpoint look like a failure
        to avoid (on the incident_exposed trips, all ten "hits" were endpoint edges).
        """
        impact = self.impact(graph, window)
        if impact is None:
            return 0
        stops = {wg.index_of(leg[0]) for leg in legs} | {wg.index_of(legs[-1][-1])} if legs else set()
        hit = 0
        for leg in legs:
            indices = [wg.index_of(v) for v in leg]
            hit += sum(1 for u, v in zip(indices, indices[1:])
                       if impact[wg.edge_id(u, v)] > 0 and not ({u, v} & stops))
        return hit


# ---------------------------------------------------------------- the baseline

class Baseline:
    """
    The thesis baseline for this evaluation: Improved ACO (Cheng 2023), run as published on
    the same subgraph and window as ANTROUTE. The algorithm itself is baseline_iaco.py,
    untouched; src/routing/baseline_router.py holds the adaptation to our data.

    What the paper does and does not define matters here, scenario by scenario:

      recommended        the paper's one optimal path -- a direct comparison.
      alternative        not defined. Section 4.2.3 Step 7 outputs "the current optimal
                         path", one route, so the baseline has no second route to pair
                         against ANTROUTE's alternative. Those trips are not evaluated
                         (shown in the app only) rather than invented for the baseline.
      multi_destination  not defined either: both of the paper's experiments are a single
                         origin to a single destination. --baseline-multi-destination
                         (default on) routes it leg by leg, the same way ANTROUTE's own
                         multi-stop routing chains single-leg routes, and the reports
                         record it as an adaptation. Off, it is left unpaired like
                         'alternative'.

    IACO plans on observed camera traffic, never on ANTROUTE's predicted risk. Both systems
    are then timed by Google Maps, which neither plans on, so neither optimises the measure
    it is judged by.
    """

    def __init__(self, graph, spatial_dir: Path, risk_edges: Path, config: IacoConfig, fallback: bool,
                 vehicle_counts: pd.DataFrame):
        coords = pd.read_csv(spatial_dir / "full_network_static_features.csv",
                             usecols=["node_id", "lat", "lon"]).set_index("node_id").reindex(graph.node_ids)
        if coords.isna().any().any():
            raise SystemExit("some subgraph nodes have no coordinates in full_network_static_features.csv")
        self.graph = IacoGraph(
            node_ids=np.asarray(graph.node_ids),
            src=graph.edge_index[0].numpy(),
            dst=graph.edge_index[1].numpy(),
            distance=edge_distances(graph),
            lat=coords["lat"].to_numpy(dtype=float),
            lon=coords["lon"].to_numpy(dtype=float),
        )
        self.free_flow = load_free_flow_seconds(spatial_dir, graph)
        self.cameras = list(graph.camera_nodes.values())
        self.node_ids = np.asarray(graph.node_ids)
        self.config = config
        self.fallback = fallback
        self.vehicle_counts = vehicle_counts
        self._observations = self._load_observations(risk_edges)
        self._inputs: Dict[str, BaselineInputs] = {}

    @staticmethod
    def _load_observations(risk_edges: Path) -> Dict[str, pd.DataFrame]:
        """Camera observations (weak_target) per window, read once."""
        header = pd.read_csv(risk_edges, nrows=0).columns
        if "weak_target" not in header:
            raise SystemExit(
                f"{risk_edges} has no weak_target column; the baseline plans on observed traffic "
                "and cannot run without it"
            )
        key = "window_start" if "window_start" in header else "window_end"
        frame = pd.read_csv(risk_edges, usecols=[key, "source_node_id", "target_node_id", "weak_target"])
        frame = frame[frame["weak_target"].notna()]
        return {str(w): rows for w, rows in frame.groupby(frame[key].astype(str))}

    def inputs_for(self, window: str) -> BaselineInputs:
        if window not in self._inputs:
            rows = self._observations.get(window)
            observed = (
                camera_observations(rows, self.node_ids, self.cameras) if rows is not None else {}
            )
            # Traffic flow n_ij: mean YOLO vehicle count per camera over this window.
            start = pd.Timestamp(window)
            flow = camera_flow(self.vehicle_counts, start, start + pd.Timedelta(minutes=WINDOW_STEPS))
            self._inputs[window] = build_inputs(self.graph, self.free_flow, observed, camera_flow=flow)
        return self._inputs[window]

    def route(self, window: str, stops: Sequence[int]) -> Tuple[Optional[List[List[int]]], str]:
        """
        (route per leg, algorithm) for this trip, or (None, reason) when it has none.

        Routed one leg at a time because Google Maps times are recorded per leg, so each leg
        needs its own node list. The algorithm is reported over the whole trip: 'iaco' only
        when IACO itself found every leg.
        """
        inputs = self.inputs_for(window)
        legs: List[List[int]] = []
        algorithms: List[str] = []
        for origin, destination in zip(stops, stops[1:]):
            try:
                leg = plan_baseline(inputs, [origin, destination], self.config, fallback=self.fallback)
            except (ValueError, KeyError) as err:
                return None, f"no route: {err}"
            legs.append(leg.nodes)
            algorithms.append(leg.algorithm)
        unique = set(algorithms)
        return legs, unique.pop() if len(unique) == 1 else "mixed"


# ---------------------------------------------------------------- template

def camera_labels(graph) -> Dict[int, str]:
    return {int(graph.node_ids[i]): label for label, i in graph.camera_nodes.items()}


def load_coordinates(spatial_dir: Path) -> Dict[int, Tuple[float, float]]:
    frame = pd.read_csv(spatial_dir / "full_network_static_features.csv", usecols=["node_id", "lat", "lon"])
    return {int(n): (float(lat), float(lon)) for n, lat, lon in frame.itertuples(index=False)}


def latlon(coords: Dict[int, Tuple[float, float]], node_id: int) -> str:
    lat, lon = coords[int(node_id)]
    return f"{lat:.6f},{lon:.6f}"


def departure_slot(window: str) -> str:
    """
    The Google Maps departure for a recorded window: its weekday and time. Google predicts
    traffic only for a present or future date, so the person doing the lookup enters the next
    date on that weekday and records it in google_departure_date.
    """
    try:
        return datetime.fromisoformat(window).strftime("%a %H:%M")
    except ValueError:
        return ""


def model_provenance(risk_edges: Path, allow_unknown: bool = False) -> dict:
    """
    Which trained model produced the risk scores, from the risk_summary.json that
    predict_congestion_risk.py writes beside risk_edges.csv: the checkpoint, whether the text
    branch (DistilBERT event text) and the node context (weather, flood, incidents, clock, road)
    reached the scores, and the per-split decoder MAE. Copied whole into metrics.json, because
    risk_edges.csv itself says none of this and the "event-aware" claim rests on it.

    Stops without risk_summary.json unless `allow_unknown`, in which case the report says the
    model is unknown rather than leaving the field out.
    """
    path = Path(risk_edges).parent / "risk_summary.json"
    if not path.exists():
        if not allow_unknown:
            raise SystemExit(
                f"{path} not found: copy it from the scoring run beside risk_edges.csv. Without it "
                "the report cannot say which checkpoint produced these scores or whether text was "
                "used (--allow-unknown-model runs anyway and records them as unknown)."
            )
        print("WARNING: no risk_summary.json; checkpoint and text use are recorded as unknown")
        return {"checkpoint": None, "use_text": None, "context": None, "risk_summary": None}
    info = json.loads(path.read_text(encoding="utf-8"))
    drop = set(info.get("context_drop") or [])
    out = {
        "checkpoint": info.get("checkpoint"),
        "use_text": info.get("use_text"),
        "context": info.get("context"),
        "context_drop": sorted(drop),
        "flood_hazard": info.get("flood_hazard"),
        "incidents_in_node_context": bool(info.get("context")) and "events" not in drop,
        "risk_summary": str(path),
        "splits": info.get("splits"),
    }
    print(f"risk scores from {out['checkpoint']}: text {'on' if out['use_text'] else 'OFF'}, "
          f"node context {'on' if out['context'] else 'OFF'}")
    if out["use_text"] is not True:
        print("WARNING: the text branch (DistilBERT event text) did not reach these risk scores")
    return out


def evaluation_windows(a) -> List[str]:
    frame = load_risk_edges(a.risk_edges)
    keys = window_key(frame)
    if a.split == "all":
        return sorted(keys.unique())
    if "split" not in frame.columns:
        raise SystemExit(f"{a.risk_edges} has no split column; pass --split all to use every window")
    windows = sorted(keys[frame["split"] == a.split].unique())
    if not windows:
        raise SystemExit(f"{a.risk_edges} has no windows in split {a.split!r}")
    return windows


def reachability(wg, cameras: Sequence[int]) -> Dict[Tuple[int, int], bool]:
    """Which ordered CCTV pairs are connected on the road graph (independent of the window)."""
    out = {}
    for d in cameras:
        remaining = remaining_cost_to(wg, wg.index_of(d))
        for o in cameras:
            out[(o, d)] = o != d and bool(np.isfinite(remaining[wg.index_of(o)]))
    return out


def forward_cost_from(wg, origin: int) -> np.ndarray:
    """
    Least dynamic cost from `origin` to every node -- the mirror of remaining_cost_to.

    Needed because a node lies on an optimal origin-to-destination route exactly when
    cost(origin -> node) + cost(node -> destination) equals cost(origin -> destination),
    and only the second term was already available.
    """
    cost = np.full(wg.n_nodes, np.inf, dtype=np.float64)
    cost[origin] = 0.0
    heap = [(0.0, int(origin))]
    while heap:
        here, u = heappop(heap)
        if here > cost[u]:
            continue
        for e in wg.out_edges(u):
            v = int(wg.dst[e])
            through = here + float(wg.weight[e])
            if through < cost[v]:
                cost[v] = through
                heappush(heap, (through, v))
    return cost


def interior_incident_pairs(wg, pairs, incident_nodes: set, tolerance: float = 1e-6):
    """
    The origin/destination pairs whose risk-only optimal route runs through an incident
    that is neither the origin nor the destination.

    This is the selection that makes the incident_exposed scenario meaningful: the layer can
    only divert a route when there is something to divert it away from, mid-route. Optimality
    is judged on the un-penalised graph on purpose -- the question is what the router would
    have done knowing nothing of the incident, which is the ablation's control arm.
    """
    if not incident_nodes:
        return []
    out = []
    remaining_cache: Dict[int, np.ndarray] = {}
    forward_cache: Dict[int, np.ndarray] = {}
    for origin, destination in pairs:
        o, d = wg.index_of(origin), wg.index_of(destination)
        if d not in remaining_cache:
            remaining_cache[d] = remaining_cost_to(wg, d)
        if o not in forward_cache:
            forward_cache[o] = forward_cost_from(wg, o)
        remaining, forward = remaining_cache[d], forward_cache[o]
        best = remaining[o]
        if not np.isfinite(best):
            continue
        for node in incident_nodes:
            if node == o or node == d:
                continue          # no route can avoid its own origin or destination
            through = forward[node] + remaining[node]
            if np.isfinite(through) and through <= best * (1.0 + tolerance):
                out.append((origin, destination))
                break
    return out


def sample_trips(cameras, connected, windows, a, rng, exposed_pairs=None) -> Dict[str, List[Tuple[str, Tuple[int, ...]]]]:
    pairs = [(o, d) for o in cameras for d in cameras if connected[(o, d)]]
    sequences = [
        s for s in permutations(cameras, a.n_stops) if all(connected[leg] for leg in zip(s, s[1:]))
    ]
    if not pairs:
        raise SystemExit("no two CCTV intersections are connected on this graph")
    if not sequences:
        raise SystemExit(f"no {a.n_stops}-stop sequence of CCTV intersections is connected; lower --n-stops")

    def draw(options):
        combos = [(w, s) for w in windows for s in options]
        picks = rng.choice(len(combos), size=min(a.n_trials, len(combos)), replace=False)
        return [combos[i] for i in sorted(picks)]

    single = draw(pairs)
    out = {RECOMMENDED: single, ALTERNATIVE: single, MULTI_DESTINATION: draw(sequences)}

    # The incident-exposed scenario draws from (window, pair) combinations already known to
    # put an incident mid-route, so its windows are not interchangeable with the others' and
    # it cannot reuse draw(). An empty list is not an error: with the event layer off, or in
    # a split whose windows carry no placeable incident, there is nothing to expose.
    combos = sorted(exposed_pairs or [])
    if combos:
        picks = rng.choice(len(combos), size=min(a.n_trials, len(combos)), replace=False)
        out[INCIDENT_EXPOSED] = [combos[i] for i in sorted(picks)]
    else:
        out[INCIDENT_EXPOSED] = []
    return out


def write_template(a, graph, baseline: "Baseline", events: "Events", config: AntColonyConfig) -> None:
    windows = evaluation_windows(a)
    coords = load_coordinates(a.spatial_dir)
    labels = camera_labels(graph)
    cameras = sorted(labels)
    rng = np.random.default_rng(a.seed)

    graphs: Dict[str, object] = {}
    plain: Dict[str, object] = {}

    def graphs_for(window):
        """ANTROUTE's graph for this window: predicted risk, plus any incidents in force."""
        if window not in graphs:
            base = weighted_graph_from_risk_file(
                graph, a.risk_edges, window=window, lam=a.lam, missing_risk=DEFAULT_MISSING_RISK
            )
            plain[window] = base
            graphs[window] = events.applied(base, graph, window)
        return graphs[window]

    # Connectivity is a property of the roads, not of today's incidents, so it is read off the
    # un-penalised graph -- an incident makes an edge expensive, never impassable.
    graphs_for(windows[0])
    connected = reachability(plain[windows[0]], cameras)
    pairs = [(o, d) for o in cameras for d in cameras if connected[(o, d)]]

    # Which (window, pair) combinations can exercise the event layer at all. Each window has
    # its own incidents and its own risk snapshot, so this is per window; windows with no
    # placeable incident contribute nothing and are skipped before the Dijkstras.
    exposed_combos: List[Tuple[str, Tuple[int, ...]]] = []
    for window in windows:
        incident_nodes = events.nodes_live(graph, window)
        if not incident_nodes:
            continue
        graphs_for(window)
        exposed_combos.extend(
            (window, pair) for pair in interior_incident_pairs(plain[window], pairs, incident_nodes)
        )
    trips = sample_trips(cameras, connected, windows, a, rng, exposed_pairs=exposed_combos)
    print(f"incident-exposed candidates: {len(exposed_combos)} (window, origin-destination) "
          f"combination(s) put an incident mid-route")

    rows = []
    routed: Dict[Tuple[str, Tuple[int, ...]], List] = {}       # one ACO run per trip
    baselines: Dict[Tuple[str, Tuple[int, ...]], Tuple] = {}   # one IACO run per trip
    algorithms: Dict[str, str] = {}                            # trip_id -> which algorithm routed the baseline
    event_rows: Dict[str, Tuple[int, int, int]] = {}           # trip_id -> (live, antroute hits, baseline hits)
    for scenario in SCENARIOS:
        for number, (window, stops) in enumerate(trips[scenario], start=1):
            wg = graphs_for(window)
            key = (window, tuple(stops))
            if key not in routed:
                routed[key] = [ant_colony_shortest_path(wg, o, d, config) for o, d in zip(stops, stops[1:])]
                baselines[key] = baseline.route(window, stops)
            # per leg: the leg's route for this scenario, or None if ACO had no alternative
            antroute_legs = [
                (r.alternatives[0].nodes if r.alternatives else None) if scenario == ALTERNATIVE else r.best.nodes
                for r in routed[key]
            ]
            # The paper outputs one optimal path, so it has no 'alternative'; multi-destination
            # is likewise undefined and is an opt-in adaptation. See the Baseline docstring.
            baseline_legs, baseline_algorithm = baselines[key]
            if scenario == ALTERNATIVE:
                baseline_legs, baseline_algorithm = None, "not defined by Cheng (2023)"
            elif scenario == MULTI_DESTINATION and not a.baseline_multi_destination:
                baseline_legs, baseline_algorithm = None, "not defined by Cheng (2023)"
            legs = {ANTROUTE: antroute_legs, BASELINE: baseline_legs or [None] * len(antroute_legs)}

            trip_id = f"{TRIP_PREFIX[scenario]}{number:02d}"
            algorithms[trip_id] = baseline_algorithm
            # Whether this trip could exercise the event layer at all, and whether each system's
            # route still runs into an incident. ANTROUTE routed with the penalty applied, the
            # baseline never sees it, so this is the per-trip evidence for RQ2.
            n_live = len(events.live(window))
            antroute_hits = (
                events.on_route(plain[window], graph, window, [r for r in antroute_legs if r])
                if n_live else 0
            )
            baseline_hits = (
                events.on_route(plain[window], graph, window, [r for r in (baseline_legs or []) if r])
                if n_live and baseline_legs else 0
            )
            event_rows[trip_id] = (n_live, antroute_hits, baseline_hits)
            for leg_no, (o, d) in enumerate(zip(stops, stops[1:]), start=1):
                routes = {s: legs[s][leg_no - 1] for s in SYSTEMS}
                row = {
                    "trip_id": trip_id,
                    "scenario_type": scenario,
                    "leg": leg_no,
                    "window": window,
                    "google_departure_slot": departure_slot(window),
                    "origin": labels[o],
                    "destination": labels[d],
                    "origin_latlon": latlon(coords, o),
                    "destination_latlon": latlon(coords, d),
                    "same_route": routes[ANTROUTE] is not None and routes[ANTROUTE] == routes[BASELINE],
                    "google_duration_seconds": "",
                    **{field: "" for field in REFERENCE_FIELDS},
                }
                row["baseline_algorithm"] = baseline_algorithm
                row["events_live"], row["antroute_event_edges"], row["baseline_event_edges"] = \
                    event_rows[trip_id]
                for s in SYSTEMS:
                    stops_on_route = pick_waypoints(wg, routes[s], a.n_waypoints) if routes[s] else []
                    row[f"{s}_waypoints"] = (
                        " | ".join(latlon(coords, v) for v in stops_on_route) if routes[s] else "NO ROUTE"
                    )
                    row[f"{s}_google_duration_seconds"] = ""
                for s in SYSTEMS:
                    row[f"{s}_route"] = " ".join(str(v) for v in routes[s]) if routes[s] else ""
                rows.append(row)
            print(f"{trip_id} {window} {' -> '.join(labels[v] for v in stops)}  baseline: {baseline_algorithm}")

    a.template.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.template, index=False)
    print(f"\nwrote {len(rows)} Google Maps lookups for {sum(len(t) for t in trips.values())} trips to {a.template}")

    # How the baseline fared is a result in its own right, so report it rather than burying it:
    # on a real road network IACO frequently finds no route at all (its ants have no way out of
    # a dead end), and that rate is what ANTROUTE's search is built to fix.
    counts = collections.Counter(algorithms.values())
    print("\nBaseline (Improved ACO, Cheng 2023) over " f"{len(algorithms)} trips:")
    for name, n in counts.most_common():
        print(f"  {n:4d}  {name}")

    # RQ2's evidence: the event layer can only be shown to work on trips where an incident was
    # actually in force. Printing the count keeps a run that happened to sample quiet windows
    # from being read as evidence that the layer does nothing.
    exposed = [t for t, (live, _, _) in event_rows.items() if live]
    print(f"\nEvent-Aware layer ({'on' if events.enabled else 'OFF'}, mu={events.mu}):")
    print(f"  {len(exposed)} of {len(event_rows)} trips had an incident in force at their window")
    if exposed:
        avoided = sum(1 for t in exposed if event_rows[t][1] == 0)
        base_hit = sum(1 for t in exposed if event_rows[t][2] > 0)
        print(f"  ANTROUTE's route is clear of every avoidable (mid-route) incident on {avoided} of those {len(exposed)}")
        print(f"  the baseline's route runs into a mid-route incident on {base_hit} of those {len(exposed)}")
        # Broken out because the first three scenarios sample origins and destinations from the
        # camera intersections, which is where incidents get matched -- so an incident there is
        # usually AT an endpoint, where avoidance is impossible and a clear route proves nothing.
        # Only the incident_exposed trips put one mid-route, so only they can show avoidance.
        mid_route = [t for t in exposed if t.startswith(TRIP_PREFIX[INCIDENT_EXPOSED])]
        if mid_route:
            clear = sum(1 for t in mid_route if event_rows[t][1] == 0)
            hit = sum(1 for t in mid_route if event_rows[t][2] > 0)
            print(f"  of the {len(mid_route)} incident_exposed trips, where the incident is "
                  f"mid-route and avoidable:")
            print(f"      ANTROUTE avoided it on {clear}")
            print(f"      the baseline ran into one on {hit}")
            print("      (incidents at a trip's own origin or destination are unavoidable and not counted)")
        else:
            print("  NOTE: no incident_exposed trips in this run, so no trip could demonstrate "
                  "avoidance -- an incident at a trip's own origin or destination cannot be routed around")
    else:
        print("  no trip in this sample is affected; --split all or more --n-trials would reach one")
    no_antroute = sum(1 for r in rows if not r["antroute_route"])
    if no_antroute:
        print(
            f"\nNOTE: {no_antroute} alternative leg(s) have NO ROUTE for ANTROUTE: its colony found no "
            "second route. Those trips cannot be paired; raise --n-iterations / --n-ants or accept fewer."
        )
    print("\nFill in the google_* columns (durations and how they were obtained), then re-run with --google-maps.")


# ---------------------------------------------------------------- scoring

def seconds(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        return float("nan")
    return float(value)


def nodes(value: str) -> Optional[List[int]]:
    return [int(v) for v in value.split()] if value and value.strip() else None


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return {
        "mean": float(x.mean()) if x.size else float("nan"),
        "sd": float(x.std(ddof=1)) if x.size > 1 else float("nan"),
    }


def citywide_conditions(window: str):
    """
    The app's risk-weighted graph for `window` over the whole city -- every road in its risk
    tier, as the app routes -- and the incident impact in force then, for trips from
    scripts/citywide_trips.py. Imported here so a sheet of camera trips only never pays the
    ~40 s whole-city build.
    """
    from app import risk_routing

    return risk_routing._weighted_graph(window), risk_routing._window_impact(window)


def reference_record(sheet: pd.DataFrame) -> dict:
    """
    How the Google Maps reference was obtained, summarised over the rows that were scored:
    access method(s), departure dates and query dates (ranges), traffic model(s), and the
    departure slots used. Mixed methods or traffic models are reported, not hidden.
    """
    def distinct(column):
        return sorted({v.strip() for v in sheet[column] if v.strip()}) if column in sheet else []

    def span(column):
        values = distinct(column)
        return {"first": values[0], "last": values[-1], "n_distinct": len(values)} if values else None

    record = {
        "reference": "Google Maps",
        "access_methods": distinct("google_access_method"),
        "traffic_models": distinct("google_traffic_model"),
        "departure_dates": span("google_departure_date"),
        "queried_on": span("google_queried_on"),
        "departure_slots": distinct("google_departure_slot"),
    }
    for key in ("access_methods", "traffic_models"):
        if len(record[key]) > 1:
            print(f"NOTE: the Google Maps lookups used {len(record[key])} different {key.replace('_', ' ')}: "
                  f"{record[key]}")
    return record


def score_sheet(a, graph, events: "Events") -> Tuple[List[dict], List[str], dict, List[dict]]:
    # Several sheets score together (the camera trips and scripts/citywide_trips.py's).
    sheet = pd.concat(
        [pd.read_csv(path, dtype=str, keep_default_na=False) for path in a.google_maps], ignore_index=True
    ).fillna("")
    if sheet.duplicated(["trip_id", "leg"]).any():
        raise SystemExit("the sheets share trip ids; each trip must appear in only one sheet")
    needed = {"trip_id", "scenario_type", "leg", "window", "google_duration_seconds", *REFERENCE_FIELDS} | {
        f"{s}_{c}" for s in SYSTEMS for c in ("route", "google_duration_seconds")
    }
    if not needed <= set(sheet.columns):
        raise SystemExit(f"{[str(x) for x in a.google_maps]} is missing {sorted(needed - set(sheet.columns))}; "
                         "make it with --template")

    # Google's own route is the same lookup for every trip with this window and leg endpoints.
    optimal_by_leg = {
        (r["window"], r["origin"], r["destination"]): r["google_duration_seconds"]
        for _, r in sheet.iterrows()
        if r["google_duration_seconds"].strip()
    }
    graphs: Dict[str, object] = {}
    trials: List[dict] = []
    notes: List[str] = []
    scored_rows: List[pd.DataFrame] = []

    for trip_id, rows in sheet.groupby("trip_id", sort=True):
        rows = rows.sort_values("leg", key=lambda c: c.astype(int))
        window, scenario = rows["window"].iloc[0], rows["scenario_type"].iloc[0]
        # Trips from scripts/citywide_trips.py start anywhere in the city: they were routed by
        # the app on the whole-city graph, and are measured on it.
        citywide = "graph" in rows and rows["graph"].iloc[0].strip() == "full"
        if citywide:
            wg, impact = citywide_conditions(window)
        else:
            if window not in graphs:
                try:
                    graphs[window] = weighted_graph_from_risk_file(
                        graph, a.risk_edges, window=window, lam=a.lam, missing_risk=DEFAULT_MISSING_RISK
                    )
                except ValueError as err:
                    raise SystemExit(f"{window}: {err}") from None
            wg, impact = graphs[window], events.impact(graph, window)

        optimal = [
            seconds(r["google_duration_seconds"].strip() or optimal_by_leg.get((r["window"], r["origin"], r["destination"])))
            for _, r in rows.iterrows()
        ]
        measured = {s: [seconds(r[f"{s}_google_duration_seconds"]) for _, r in rows.iterrows()] for s in SYSTEMS}
        same = [str(r.get("same_route", "")).strip().lower() == "true" for _, r in rows.iterrows()]
        for i, is_same in enumerate(same):
            if is_same:
                shared = next((measured[s][i] for s in SYSTEMS if np.isfinite(measured[s][i])), float("nan"))
                for s in SYSTEMS:
                    measured[s][i] = shared

        routes = {s: [nodes(r[f"{s}_route"]) for _, r in rows.iterrows()] for s in SYSTEMS}
        if any(leg is None for leg in routes[ANTROUTE]):
            notes.append(f"{trip_id}: ANTROUTE has no route for some leg; trip left out")
            continue
        # Only paired trips are evaluated. Without a baseline route (Cheng defines no
        # alternative, or IACO found none) the trip is left out: alternatives are shown in the
        # app, not evaluated.
        if any(leg is None for leg in routes[BASELINE]):
            why = rows["baseline_algorithm"].iloc[0] if "baseline_algorithm" in rows else "no route"
            notes.append(f"{trip_id}: baseline has no route ({why}); not evaluated")
            continue
        systems = SYSTEMS
        if not all(np.isfinite(v) for v in optimal + [x for s in systems for x in measured[s]]):
            notes.append(f"{trip_id}: Google Maps times not all filled in; skipped")
            continue
        unrecorded = [f for f in REFERENCE_FIELDS if any(not str(v).strip() for v in rows[f])]
        if unrecorded:
            notes.append(f"{trip_id}: Google Maps lookup not documented ({', '.join(unrecorded)} blank); skipped")
            continue
        scored_rows.append(rows)
        labels = list(rows["origin"]) + [rows["destination"].iloc[-1]]
        for s in systems:
            trial = score_trip(wg, routes[s], s, scenario, optimal, measured[s], trip_id, impact=impact)
            trial["stop_labels"] = labels
            trial["graph"] = "full" if citywide else "camera_subgraph"
            trial["risk_sources"] = route_provenance(wg, trial["path"], citywide)
            trial["route_optimality"] = float(route_optimality([trial["c_optimal"]], [trial["c_predicted"]])[0])
            trials.append(trial)
    reference = reference_record(pd.concat(scored_rows, ignore_index=True)) if scored_rows else {}
    return trials, notes, reference


def route_provenance(wg, path: Sequence[int], citywide: bool) -> Dict[str, float]:
    """
    Share of the route's length in each risk tier: observed (scored by the model at a camera),
    propagated (borrowed from the nearest scored road within 1.5 km) and assumed (the non-visual
    estimate, or the window median when no estimate exists -- the default). The camera subgraph
    is scored everywhere, so a route there is entirely observed.
    """
    if not citywide:
        return {"observed": 1.0, "propagated": 0.0, "assumed": 0.0}
    from app import risk_routing
    return risk_routing.risk_sources(wg, [int(v) for v in path])


def describe(stops: Sequence[int], labels: Dict[int, str]) -> str:
    return " -> ".join(labels.get(v, str(v)) for v in stops)


def stops_text(trial: dict, labels: Dict[int, str]) -> str:
    """A trial's stops by name: the sheet's own labels (citywide trips start at any road), else
    the camera intersection names."""
    if trial.get("stop_labels"):
        return " -> ".join(trial["stop_labels"])
    return describe(trial["stops"], labels)


def per_trip_comparison(trip_ids, scenario_of, by, labels) -> List[dict]:
    """
    One row per trip: ANTROUTE and the baseline side by side on every tested metric (Route
    Optimality, congestion exposure, event exposure), the relative difference by Equation 8
    (higher is better) or Equation 7 (lower is better), and which system did better on that
    trip; route length beside them, descriptive only. Positive Δ% means ANTROUTE is better.
    """
    rows = []
    for i in trip_ids:
        ant, base = by[(i, ANTROUTE)], by[(i, BASELINE)]
        row = {
            "scenario_type": scenario_of[i],
            "trip_id": i,
            "window": ant["window"],
            "graph": ant.get("graph", "camera_subgraph"),
            "stops": stops_text(ant, labels),
            # Graph node ids, origin first: what the app matches a planned trip against.
            "stop_nodes": " ".join(str(v) for v in ant["stops"]),
            "legs": len(ant["reference_route_seconds"]),
        }
        for system, trial in ((ANTROUTE, ant), (BASELINE, base)):
            for tier, share in (trial.get("risk_sources") or {}).items():
                row[f"{tier}_share_{system}"] = share
        wins = {ANTROUTE: 0, BASELINE: 0}
        for metric in TESTED_METRICS:
            p, b = float(ant[metric]), float(base[metric])
            higher = metric in HIGHER_IS_BETTER
            row[f"{metric}_antroute"] = p
            row[f"{metric}_baseline"] = b
            if not (math.isfinite(p) and math.isfinite(b)):
                row[f"{metric}_delta_pct"], row[f"{metric}_better"] = float("nan"), ""
                continue
            row[f"{metric}_delta_pct"] = relative_difference(b, p, higher_is_better=higher)
            if math.isclose(p, b, rel_tol=1e-9, abs_tol=1e-9):
                row[f"{metric}_better"] = "tie"
            else:
                winner = ANTROUTE if (p > b) == higher else BASELINE
                row[f"{metric}_better"] = winner
                wins[winner] += 1
        for metric in DESCRIPTIVE_METRICS:
            row[f"{metric}_antroute"], row[f"{metric}_baseline"] = float(ant[metric]), float(base[metric])
        row["metrics_won_antroute"], row["metrics_won_baseline"] = wins[ANTROUTE], wins[BASELINE]
        rows.append(row)
    return rows


def write_reports(a, graph, trials: List[dict], provenance: dict) -> dict:
    labels = camera_labels(graph)
    by = {(t["trip_id"], t["system"]): t for t in trials}
    trip_ids = sorted({t["trip_id"] for t in trials})
    scenario_of = {t["trip_id"]: t["scenario_type"] for t in trials}
    groups = {"overall": trip_ids}
    groups.update({sc: [i for i in trip_ids if scenario_of[i] == sc] for sc in SCENARIOS})
    groups = {k: v for k, v in groups.items() if v}

    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    flat = []
    for t in trials:
        row = dict(t)
        row["stops"] = stops_text(t, labels)
        row.pop("stop_labels", None)
        row["path"] = " ".join(str(v) for v in t["path"])
        row["reference_route_seconds"] = " | ".join(f"{v:g}" for v in t["reference_route_seconds"])
        flat.append(row)
    pd.DataFrame(flat).to_csv(out / "trials.csv", index=False)

    def per_trial(columns, name):
        rows = [
            {"scenario_type": scenario_of[i], "trial": i, "system": s, **{c: by[(i, s)][c] for c in columns}}
            for i in trip_ids for s in SYSTEMS
        ]
        pd.DataFrame(rows).to_csv(out / name, index=False)

    per_trial(("c_optimal", "c_predicted", "route_optimality"), "appendix1_route_optimality.csv")
    per_trial(("congestion_exposure", "event_exposure", "avoidable_event_exposure", "distance_m"), "appendix2_exposure.csv")
    pd.DataFrame(
        [
            {
                "scenario_type": scenario_of[i],
                "trial": i,
                "model": s,
                "path": stops_text(by[(i, s)], labels),
                "hops": by[(i, s)]["hops"],
                "total_distance_m": by[(i, s)]["distance_m"],
                "congestion_exposure": by[(i, s)]["congestion_exposure"],
                "event_exposure": by[(i, s)]["event_exposure"],
                "avoidable_event_exposure": by[(i, s)]["avoidable_event_exposure"],
            }
            for i in trip_ids for s in (BASELINE, ANTROUTE)
        ]
    ).to_csv(out / "appendix3_routing_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_type": scenario_of[i],
                "trial": i,
                "proposed": by[(i, ANTROUTE)]["route_optimality"],
                "baseline": by[(i, BASELINE)]["route_optimality"],
                "d_i": by[(i, BASELINE)]["route_optimality"] - by[(i, ANTROUTE)]["route_optimality"],
                "relative_difference_pct": relative_difference(
                    by[(i, BASELINE)]["route_optimality"], by[(i, ANTROUTE)]["route_optimality"],
                    higher_is_better=True,
                ),
            }
            for i in trip_ids
        ]
    ).to_csv(out / "appendix3_route_optimality_pairs.csv", index=False)

    per_trip = per_trip_comparison(trip_ids, scenario_of, by, labels)
    pd.DataFrame(per_trip).to_csv(out / "per_trip.csv", index=False)

    significance, summary = [], {}
    for group, ids in groups.items():
        summary[group] = {}
        tested = {}
        for metric in TESTED_METRICS:
            proposed = [by[(i, ANTROUTE)][metric] for i in ids]
            baseline = [by[(i, BASELINE)][metric] for i in ids]
            c = compare_paired(metric, proposed, baseline)
            tested[metric] = c
            summary[group][metric] = {ANTROUTE: mean_sd(proposed), BASELINE: mean_sd(baseline)}
        holm_adjust([tested[m] for m in Q3_FAMILY if m in tested])        # Q3: congestion + event, Holm
        holm_adjust([c for m, c in tested.items() if m not in Q3_FAMILY and m != "avoidable_event_exposure"])
        holm_adjust([tested[m] for m in ("avoidable_event_exposure",) if m in tested])
        significance.extend({"scenario_type": group, **asdict(c)} for c in tested.values())
        for metric in DESCRIPTIVE_METRICS:
            summary[group][metric] = {s: mean_sd([by[(i, s)][metric] for i in ids]) for s in SYSTEMS}
    pd.DataFrame(significance).to_csv(out / "appendix3_significance.csv", index=False)

    pooled = {s: compute_metrics_by_scenario([t for t in trials if t["system"] == s]) for s in SYSTEMS}
    report = {
        "scope": "Metro Manila",
        "risk_edges": str(a.risk_edges),
        "reference_sheets": [str(x) for x in a.google_maps],
        "reference": provenance["reference"],
        "model": provenance["model"],
        "metrics": {
            "tested": list(TESTED_METRICS),
            "descriptive": list(DESCRIPTIVE_METRICS),
            "not_evaluated": "ETA -- not a research question or metric of this thesis",
        },
        ANTROUTE: {
            "algorithm": "ACO on the risk-weighted graph with the Event-Aware layer",
            "lambda": a.lam,
            "event_layer": describe_event_layer(0.0 if a.no_events else a.event_mu),
        },
        BASELINE: {
            "algorithm": "Improved ACO (Cheng 2023), doi:10.1155/2023/7651100",
            "alpha": a.baseline_alpha,
            "beta": a.baseline_beta,
            "rho": a.baseline_rho,
            "q0": a.baseline_q0,
            "n_ants": a.n_ants,
            "n_iterations": a.n_iterations,
            "shortest_distance_fallback": a.baseline_fallback,
            "flow_term": "mean YOLO vehicle count per camera per window (auto_labels.csv)",
            "dead_end_recovery": True,
            "multi_destination_adapted": a.baseline_multi_destination,
            "plans_on": "observed camera congestion (weak_target), not ANTROUTE's predicted risk",
        },
        "per_trip": per_trip,
        "per_trial_mean_sd": summary,
        "pooled": {s: {k: asdict(m) for k, m in r.items()} for s, r in pooled.items()},
        "significance": significance,
    }
    (out / "metrics.json").write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def score(a, graph, events: "Events", model: dict) -> None:
    trials, notes, reference = score_sheet(a, graph, events)
    for note in notes:
        print(f"NOTE: {note}")
    if not trials:
        raise SystemExit("the Google Maps sheet(s) have no fully filled-in, documented trips")
    over = sum(t["c_optimal"] > t["c_predicted"] for t in trials)
    if over:
        print(
            f"NOTE: {over} trial(s) score above 100%: Google timed the forced route faster than its own. "
            "Usually traffic shifted between lookups -- re-check those rows."
        )

    report = write_reports(a, graph, trials, {"reference": reference, "model": model})
    sig = {(r["scenario_type"], r["metric"]): r for r in report["significance"]}
    print(f"\n{'scenario':<18}{'n':>3}  {'optimality % ant / base':>24}  {'Wilcoxon p':>10}  "
          f"{'congestion exp. ant / base':>27}  {'event exp. ant / base':>22}")
    for group, metrics in report["per_trial_mean_sd"].items():
        ro, ce, ee = metrics["route_optimality"], metrics["congestion_exposure"], metrics["event_exposure"]
        p = sig[(group, "route_optimality")]["wilcoxon_p"]
        print(
            f"{group:<18}{sig[(group, 'route_optimality')]['n']:>3}  "
            f"{ro[ANTROUTE]['mean']:>10.2f} / {ro[BASELINE]['mean']:<11.2f}  {p:>10.4f}  "
            f"{ce[ANTROUTE]['mean']:>12.2f} / {ce[BASELINE]['mean']:<12.2f}  "
            f"{ee[ANTROUTE]['mean']:>9.2f} / {ee[BASELINE]['mean']:<9.2f}"
        )
    print(f"\n{'trip':<5} {'scenario':<18}{'optimality % ant / base':>24}  {'congestion ant / base':>22}  "
          f"{'event ant / base':>17}  won ant:base")
    for r in report["per_trip"]:
        print(
            f"{r['trip_id']:<5} {r['scenario_type']:<18}"
            f"{r['route_optimality_antroute']:>10.1f} / {r['route_optimality_baseline']:<11.1f}  "
            f"{r['congestion_exposure_antroute']:>9.2f} / {r['congestion_exposure_baseline']:<10.2f}  "
            f"{r['event_exposure_antroute']:>7.2f} / {r['event_exposure_baseline']:<7.2f}  "
            f"{r['metrics_won_antroute']}:{r['metrics_won_baseline']}"
        )
    print(f"  {scenario:18s} n={m['n']:3d}  optimality {m['route_optimality']['mean']:.1f}%  "
                  f"congestion {m['congestion_exposure']['mean']:.2f}  event {m['event_exposure']['mean']:.2f}")
    print(f"\nwrote appendix tables, per_trip.csv, trials.csv and metrics.json to {a.out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--risk-edges", required=True, type=Path)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--template", type=Path, help="route the test trips and write the Google Maps sheet")
    mode.add_argument("--google-maps", type=Path, nargs="+",
                      help="the filled-in sheet(s) from --template (and scripts/citywide_trips.py), to score")
    p.add_argument("--allow-unknown-model", action="store_true",
                   help="run without risk_summary.json; the report then records the checkpoint as unknown")
    p.add_argument("--spatial-dir", type=Path, default=REPO_ROOT / "data/processed/spatial")
    p.add_argument("-k", type=int, default=DEFAULT_K)
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA, help="ANTROUTE risk penalty")
    # The Event-Aware layer is part of the method and on for evaluation; --no-events is the ablation.
    p.add_argument("--event-mu", type=float, default=DEFAULT_MU,
                   help="how hard ANTROUTE avoids reported incidents; must be > 0 (use --no-events to ablate)")
    p.add_argument("--no-events", action="store_true",
                   help="ablation: route with predicted risk only. Incidents are still parsed and "
                        "still select the incident_exposed trips and measure event exposure, so both "
                        "arms study the same trips; the router simply never sees the penalty")
    # The baseline is Improved ACO (Cheng 2023). Defaults are the paper's own (sec. 5.1.2);
    # q0 = 0.7 is its "large network" setting, and the global volatility xi it never states
    # follows baseline_iaco's default.
    p.add_argument("--baseline-alpha", type=float, default=1.0)
    p.add_argument("--baseline-beta", type=float, default=5.0)
    p.add_argument("--baseline-rho", type=float, default=0.3)
    p.add_argument("--baseline-q0", type=float, default=0.7)
    p.add_argument(
        "--baseline-fallback", action="store_true",
        help="give a trip IACO cannot route the spatial shortest-distance route (Cheng's Table 9 "
             "comparator) instead of leaving it unpaired. Off by default: a shortest-distance "
             "route is not the Improved ACO baseline the thesis compares against",
    )
    p.add_argument(
        "--auto-labels", type=Path, nargs="+",
        help="--template: auto_labels.csv file(s) from the labeling step; their YOLO vehicle counts "
             "are the traffic flow term of the baseline's cost (Cheng eq. 6)",
    )
    p.add_argument("--camera-map", type=Path, default=REPO_ROOT / "configs/camera_nodes.csv")
    p.add_argument(
        "--no-baseline-multi-destination", dest="baseline_multi_destination", action="store_false",
        help="Cheng (2023) routes a single origin to a single destination. By default the "
             "baseline is adapted to multi-stop trips leg by leg; this leaves them unpaired instead",
    )
    p.add_argument("--split", default="test", help='--template: risk-file split to draw windows from, or "all"')
    p.add_argument("--n-trials", type=int, default=10, help="--template: trials per scenario")
    p.add_argument("--n-stops", type=int, default=3, help="--template: stops per multi-destination trip")
    p.add_argument("--n-waypoints", type=int, default=6,
                   help="--template: Google Maps stops per leg (the web app takes up to 9 waypoints)")
    p.add_argument("--n-ants", type=int, default=20)
    p.add_argument("--n-iterations", type=int, default=60)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--evaporation", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs/routing_eval")
    a = p.parse_args()
    if a.n_stops < 3:
        p.error("--n-stops must be at least 3 (origin, one intermediate stop, destination)")
    if not a.no_events and not a.event_mu > 0:
        p.error("--event-mu must be > 0: the Event-Aware layer is part of the method (ablate with --no-events)")
    if a.template and not a.auto_labels:
        p.error("--template needs --auto-labels: the baseline's traffic-flow term is built from them")

    model = model_provenance(a.risk_edges, allow_unknown=a.allow_unknown_model)
    graph = build_subgraph(a.spatial_dir, a.k)
    events = Events(graph, a.spatial_dir, mu=a.event_mu, enabled=not a.no_events)
    if a.template:
        baseline = Baseline(
            graph,
            a.spatial_dir,
            a.risk_edges,
            IacoConfig(
                n_ants=a.n_ants,
                n_iterations=a.n_iterations,
                alpha=a.baseline_alpha,
                beta=a.baseline_beta,
                rho=a.baseline_rho,
                q0=a.baseline_q0,
                seed=a.seed,
            ),
            fallback=a.baseline_fallback,
            vehicle_counts=load_vehicle_counts(a.auto_labels, graph.camera_nodes, a.camera_map),
        )
        config = AntColonyConfig(
            n_ants=a.n_ants,
            n_iterations=a.n_iterations,
            alpha=a.alpha,
            beta=a.beta,
            evaporation=a.evaporation,
            n_alternatives=1,
            seed=a.seed,
        )
        write_template(a, graph, baseline, events, config)
    else:
        score(a, graph, events, model)


if __name__ == "__main__":
    main()
