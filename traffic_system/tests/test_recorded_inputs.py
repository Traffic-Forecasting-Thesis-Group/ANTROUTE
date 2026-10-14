"""
The recorded inputs panel, per-route fields, simulation mode and the what-if table.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app import recorded, risk_routing as rr
from app.what_if import WhatIfUnavailableError, load_what_if
from test_risk_routing import make_network


def frame(split, labelled=False):
    return pd.DataFrame({
        "source_node_id": [1, 2], "target_node_id": [2, 3], "risk": [0.4, 0.6],
        "weak_target": [1.0, 0.5] if labelled else [np.nan, np.nan], "split": [split, split],
    })


def two_day_network():
    net = make_network({1: (14.600, 121.000), 2: (14.601, 121.000), 3: (14.620, 121.020)})
    net.risk_by_window = {
        "2026-05-22T07:00:00": frame("val", True),
        "2026-05-25T07:00:00": frame("test", True),
        "2026-05-25T17:20:00": frame("test", True),
        "2026-05-15T07:00:00": frame("infer"),
    }
    return net


def test_a_human_label_is_light_medium_or_heavy():
    assert [recorded._label(s) for s in (0.0, 0.5, 1.0)] == ["Light", "Medium", "Heavy"]
    assert recorded._label(None) is None and recorded._label(float("nan")) is None


def test_recorded_sessions_flag_only_the_test_day_as_held_out(monkeypatch):
    monkeypatch.setattr(rr, "_network", two_day_network)
    sessions = {(s["day"], s["session"]): s for s in recorded.recorded_windows()}
    assert sessions[("2026-05-25", "AM")]["held_out"] and sessions[("2026-05-25", "PM")]["held_out"]
    assert not sessions[("2026-05-22", "AM")]["held_out"] and sessions[("2026-05-22", "AM")]["split"] == "val"
    assert sessions[("2026-05-15", "AM")]["split"] == "infer" and sessions[("2026-05-15", "AM")]["labelled_edges"] == 0
    assert "never saw" in sessions[("2026-05-25", "PM")]["note"]
    window = sessions[("2026-05-25", "PM")]["windows"][0]
    assert datetime.fromisoformat(window["end"]) - datetime.fromisoformat(window["start"]) == timedelta(minutes=30)
    assert list(sessions) == sorted(sessions)                                  # in date order


def test_the_window_description_says_which_split_it_is_in(monkeypatch):
    monkeypatch.setattr(rr, "_network", two_day_network)
    info = recorded.window_description("2026-05-25T17:20:00")
    assert info["held_out"] and info["split"] == "test" and info["session"] == "PM"
    assert info["label"] == "Mon May 25, 5:20 PM - 5:50 PM"
    assert not recorded.window_description("2026-05-22T07:00:00")["held_out"]


def test_a_camera_or_incident_counts_as_on_the_route_within_200_m(monkeypatch):
    net = make_network({1: (14.6000, 121.0000), 2: (14.6100, 121.0000)})
    on = {"camera": "on", "lat": 14.6000, "lon": 121.0010}                      # ~ 108 m east of node 1
    far = {"camera": "far", "lat": 14.6000, "lon": 121.0100}                    # ~ 1.1 km east
    out = recorded._near_route(net, [1, 2], [on, far, {"camera": "unplaced"}])
    assert [t["camera"] for t in out] == ["on"] and 90 < out[0]["distance_m"] < 130
    assert recorded._near_route(net, [1, 2], []) == []


def test_route_inputs_are_none_without_a_recorded_window():
    class Graph:
        window = None
    assert recorded.route_inputs(two_day_network(), Graph(), [1, 2]) is None


def test_the_what_if_table_follows_the_checkpoint_the_routes_use(tmp_path, monkeypatch):
    from app import what_if
    from app.config import settings

    (tmp_path / "v8.json").write_text(json.dumps({"model": "v8", "windows": 24, "note": "n", "cases": []}), encoding="utf-8")
    (tmp_path / "v9.json").write_text(json.dumps({"model": "v9", "windows": 24, "note": "n", "cases": []}), encoding="utf-8")
    monkeypatch.setattr(settings, "what_if_dir", str(tmp_path))
    monkeypatch.setattr(what_if, "_checkpoint_stem", lambda: "v9")
    assert load_what_if()["model"] == "v9"
    monkeypatch.setattr(what_if, "_checkpoint_stem", lambda: "v10")
    stale = load_what_if()
    assert "rebuild it" in stale["note"]                    # a table for another checkpoint says so
    monkeypatch.setattr(settings, "what_if_dir", str(tmp_path / "nothing"))
    with pytest.raises(WhatIfUnavailableError):
        load_what_if()


def test_simulation_plans_on_the_chosen_window_and_refuses_an_unrecorded_one(monkeypatch):
    from app import route_engine

    monkeypatch.setattr(rr, "_network", two_day_network)
    window, note = route_engine.simulated_window("2026-05-25T17:20:00")
    assert window == "2026-05-25T17:20:00"
    assert note.startswith("Simulation:") and "held out" in note
    assert "held out" not in route_engine.simulated_window("2026-05-22T07:00:00")[1]
    with pytest.raises(route_engine.UnknownWindowError):
        route_engine.simulated_window("2026-05-26T17:20:00")
