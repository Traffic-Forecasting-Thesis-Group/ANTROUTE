"""
Hourly weather (weatherstack_hourly.csv) and the OSM road attributes: each minute gets the
weather of its own hour, a checkpoint can never be fed weather of the other resolution, and the
decoder's road inputs are versioned so older checkpoints keep loading.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from src.data import node_context
from src.data.alignment import SESSION_STEPS, build_sessions, session_start
from src.data.training_data import check_weather_resolution, load_weather_daily, weather_resolution

DAY = date(2026, 5, 18)


def hourly_csv(tmp_path, rows):
    path = tmp_path / "weatherstack_hourly.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_the_resolution_is_read_from_the_file(tmp_path):
    hourly = hourly_csv(tmp_path, [{"date": "2026-05-18", "hour": 7, "grid_cell": "a", "ws_temp_c": 28,
                                    "ws_precip_mm": 0.0, "ws_humidity_pct": 70}])
    daily = tmp_path / "daily.csv"
    pd.DataFrame([{"date": "2026-05-18", "grid_cell": "a", "ws_temp_c": 35, "ws_precip_mm": 7.2,
                   "ws_humidity_pct": 67}]).to_csv(daily, index=False)
    assert weather_resolution(hourly) == "hourly" and weather_resolution(daily) == "daily"


def test_a_checkpoint_refuses_weather_of_the_other_resolution(tmp_path):
    hourly = hourly_csv(tmp_path, [{"date": "2026-05-18", "hour": 7, "grid_cell": "a", "ws_temp_c": 28,
                                    "ws_precip_mm": 0.0, "ws_humidity_pct": 70}])
    with pytest.raises(ValueError, match="trained on daily"):
        check_weather_resolution({}, hourly)                    # no key: an older, daily checkpoint
    check_weather_resolution({"weather_resolution": "hourly"}, hourly)


def test_hourly_city_weather_is_averaged_per_hour_not_per_day(tmp_path):
    path = hourly_csv(tmp_path, [
        {"date": "2026-05-18", "hour": h, "grid_cell": c, "ws_temp_c": t, "ws_precip_mm": p, "ws_humidity_pct": 60}
        for h, c, t, p in [(7, "a", 28, 0.0), (7, "b", 30, 0.0), (17, "a", 32, 2.0), (17, "b", 34, 4.0)]
    ])
    frame = load_weather_daily(path)
    assert list(frame.columns[:2]) == ["date", "hour"] and len(frame) == 2
    assert frame.set_index("hour").loc[17, "ws_precip_mm"] == pytest.approx(3.0)


def test_each_minute_of_a_session_gets_its_own_hours_weather():
    weather = pd.DataFrame([
        {"date": DAY, "hour": h, "ws_temp_c": 20.0 + h, "ws_precip_mm": 0.0, "ws_humidity_pct": 50.0}
        for h in range(24)
    ])
    frames = pd.DataFrame({"camera_id": ["cam"], "timestamp": [session_start(DAY, "PM")], "frame_path": ["f.jpg"]})
    tweets = pd.DataFrame({"camera_id": pd.Series(dtype=str), "created_at": pd.Series(dtype="datetime64[ns]"),
                           "embedding": pd.Series(dtype=object)})
    records, scaler, _ = build_sessions(frames, tweets, weather, ["cam"], ["ws_temp_c", "ws_precip_mm", "ws_humidity_pct"],
                                        visual_split={(DAY, "PM"): "train"})
    record = next(r for r in records if r.session == "PM")
    assert record.weather.shape == (SESSION_STEPS, 3)
    start = session_start(DAY, "PM")
    temps = record.weather[:, 0] * (scaler.max_[0] - scaler.min_[0]) + scaler.min_[0]
    for step in (0, 59, 60, 119):
        assert temps[step] == pytest.approx(20.0 + (start + pd.Timedelta(minutes=step)).hour)


def test_node_weather_reads_the_hour_and_falls_back_to_the_days_mean():
    ctx = node_context.NodeContext(
        static=np.zeros((2, node_context.N_FEATURES), np.float32), flood=np.zeros(2, np.float32),
        cells=np.array(["a", "b"]), weather={(DAY, 7, "a"): np.array([1.0, 0.0, 0.0]),
                                             (DAY, 8, "a"): np.array([0.0, 1.0, 0.0])},
        weather_min=np.zeros(3), weather_max=np.ones(3), hourly=True,
    )
    assert ctx.node_weather(DAY, 7)[0] == pytest.approx([1.0, 0.0, 0.0])
    assert ctx.node_weather(DAY, 7)[1] == pytest.approx([1.0, 0.0, 0.0])     # cell b: that hour's citywide mean
    assert ctx.node_weather(DAY, 17)[0] == pytest.approx([0.5, 0.5, 0.0])    # no row at 17:00: the day's mean
    assert ctx.config()["weather_resolution"] == "hourly"
    out = ctx.at_times([datetime(2026, 5, 18, 7, 59), datetime(2026, 5, 18, 8, 0)]).numpy()
    temp = node_context.COLUMN["temp"]
    assert out[0, 0, temp] == pytest.approx(1.0) and out[1, 0, temp] == pytest.approx(0.0)


def test_road_class_ranks_through_roads_above_side_streets():
    rc = node_context.ROAD_CLASS
    assert rc["motorway"] > rc["trunk"] > rc["primary"] > rc["secondary"] > rc["tertiary"] > rc["residential"]
    assert rc["primary"] > rc["primary_link"] > rc["secondary"]


def test_older_checkpoints_keep_their_two_road_inputs():
    assert node_context.EDGE_FEATURES_V1 == ["log_length", "free_flow_speed"]
    assert node_context.EDGE_FEATURES_V2[:2] == node_context.EDGE_FEATURES_V1
    with pytest.raises(ValueError, match="unknown edge feature"):
        node_context.edge_features(None, None, ["lane_width"])


def test_sources_are_marked_before_lanes_are_filled_in():
    import networkx as nx
    from src.ingestion.spatial_topology import add_lane_counts, mark_sources

    G = nx.MultiDiGraph()
    G.add_edge(1, 2, highway="primary", lanes="3", maxspeed="60")
    G.add_edge(2, 3, highway="residential")
    G = add_lane_counts(mark_sources(G))
    tagged, assumed = G.edges[1, 2, 0], G.edges[2, 3, 0]
    assert (tagged["lanes_source"], tagged["maxspeed_source"], tagged["lanes"]) == ("osm", "osm", 3)
    assert (assumed["lanes_source"], assumed["maxspeed_source"], assumed["lanes"]) == ("road_type", "road_type", 1)


def test_road_attributes_read_lanes_and_road_type_from_the_real_table():
    from pathlib import Path

    from src.data.graph_data import build_subgraph

    spatial = Path(__file__).resolve().parents[1] / "data/processed/spatial"
    if not (spatial / "full_network_edge_features.csv").exists():
        pytest.skip("full_network_edge_features.csv not built")
    graph = build_subgraph(spatial, 8)
    attrs = node_context.road_attributes(graph, spatial)
    assert attrs["lanes"].shape == attrs["road_class"].shape == (graph.edge_index.shape[1],)
    assert set(np.round(attrs["road_class"], 2)) <= {round(v, 2) for v in node_context.ROAD_CLASS.values()}
    assert attrs["lanes"].max() > 1 / node_context.MAX_LANES        # EDSA has more than one lane
    assert node_context.edge_features(graph, spatial, node_context.EDGE_FEATURES_V2).shape[1] == 4


def test_missing_days_are_requested_in_runs_of_at_most_60_consecutive_days():
    from datetime import timedelta

    from src.ingestion.temporal_weather import contiguous_chunks

    start = date(2025, 6, 1)
    days = [start + timedelta(days=i) for i in range(130)]
    recorded = {date(2025, 7, 1), date(2025, 7, 2)}          # already fetched: the run splits around them
    chunks = contiguous_chunks([d for d in days if d not in recorded])
    assert chunks[0] == (date(2025, 6, 1), date(2025, 6, 30))
    assert all((end - s).days < 60 for s, end in chunks)
    covered = {s + timedelta(days=i) for s, end in chunks for i in range((end - s).days + 1)}
    assert covered == set(days) - recorded
