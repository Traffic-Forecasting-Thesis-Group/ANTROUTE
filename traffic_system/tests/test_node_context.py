"""Per-node context (src/data/node_context.py) and the multimodal CRS it feeds."""

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from src.data.graph_data import subgraph_from_arrays
from src.data.node_context import (
    COLUMN,
    EVENT_TTL_MINUTES,
    FEATURES,
    GROUPS,
    N_FEATURES,
    NodeContext,
    edge_features,
    event_node_index,
    group_columns,
)
from src.models.cnn_lstm_fusion import CNNLSTMFusion
from src.models.mlp_decoder import MLPDecoder
from src.models.radr_stgnn import RADRSTGNN
from src.models.source_sensitivity import SOURCES, constant_edges, perturb, source_sensitivity
from src.models.traffic_risk_model_edge import TrafficRiskModel, camera_edge_ids, edge_risk_loss
from src.routing.event_layer import TrafficEvent

SIDE = 6                       # 6 x 6 road grid
AVENUE_ROW = 2                 # a 60 km/h avenue; every other road is 20 km/h
CAMERAS = {"C0": 0, "C1": 3, "C2": 14, "C3": 20, "C4": 33, "C5": 35}
FLOOD = [0, 1, 2, 3] * 9       # node i -> level i % 4
DAY = date(2026, 5, 4)         # a Monday
T = 30


def node_id(i):
    return 1000 + i


def coords(i):
    r, c = divmod(i, SIDE)
    return 14.50 + r * 0.02, 121.00 + c * 0.02   # spans three 0.05-degree weather rows


@pytest.fixture(scope="module")
def city(tmp_path_factory):
    d = tmp_path_factory.mktemp("spatial")
    n = SIDE * SIDE
    rows, cols, length, seconds = [], [], [], []
    for i in range(n):
        r, c = divmod(i, SIDE)
        for j in ([i + 1] if c < SIDE - 1 else []) + ([i + SIDE] if r < SIDE - 1 else []):
            kmh = 60.0 if r == AVENUE_ROW and j == i + 1 else 20.0
            for u, v in ((i, j), (j, i)):
                rows.append(u); cols.append(v); length.append(100.0); seconds.append(100.0 / (kmh / 3.6))
    adjacency = sp.csr_matrix((length, (rows, cols)), shape=(n, n))
    sp.save_npz(d / "metro_manila_adjacency.npz", adjacency)
    sp.save_npz(d / "metro_manila_travel_time.npz", sp.csr_matrix((seconds, (rows, cols)), shape=(n, n)))
    order = np.array([node_id(i) for i in range(n)])
    np.save(d / "metro_manila_node_order.npy", order)
    camera_of = {v: k for k, v in CAMERAS.items()}
    pd.DataFrame({
        "node_id": order,
        "lat": [coords(i)[0] for i in range(n)],
        "lon": [coords(i)[1] for i in range(n)],
        "is_cctv_node": [i in camera_of for i in range(n)],
        "cctv_label": [camera_of.get(i, "") for i in range(n)],
        "flood_hazard_level": FLOOD,
    }).to_csv(d / "full_network_static_features.csv", index=False)

    cells = sorted({f"{round(coords(i)[0] / 0.05)}_{round(coords(i)[1] / 0.05)}" for i in range(n)})
    weather = []
    for k, cell in enumerate(cells):
        for offset, rain in ((0, 0.0), (1, 10.0)):   # a dry day and a wet day per cell
            weather.append({"date": (DAY + timedelta(days=offset)).isoformat(), "ws_temp_c": 30 + k,
                            "ws_precip_mm": rain + k, "ws_humidity_pct": 60 + 5 * k, "grid_cell": cell})
    pd.DataFrame(weather).to_csv(d / "weather.csv", index=False)
    graph = subgraph_from_arrays(adjacency, order, dict(CAMERAS), k=SIDE * 2)
    return {"dir": d, "graph": graph, "cells": cells}


def build(city, **kw):
    kw.setdefault("train_days", [DAY, DAY + timedelta(days=1)])
    return NodeContext.build(city["graph"], city["dir"], city["dir"] / "weather.csv", **kw)


def event(at, intersection="C2", lanes=2):
    return TrafficEvent(at=at, kind="crash", location_text="x", landmark="x",
                        intersection=intersection, lanes_occupied=lanes, direction=None)


# ---------------------------------------------------------------- the context itself
def test_every_source_has_its_columns():
    assert N_FEATURES == len(FEATURES) == sum(len(v) for v in GROUPS.values())
    assert set(GROUPS) == {"weather", "flood", "events", "temporal", "spatial"}


def test_weather_is_daily_and_per_grid_cell(city):
    ctx = build(city)
    dry, wet = ctx.window(DAY, "AM", 0), ctx.window(DAY + timedelta(days=1), "AM", 0)
    precip = COLUMN["precip"]
    # one value per day: constant over the window's minutes
    assert torch.equal(dry[0, :, precip], dry[-1, :, precip])
    # each node gets its own cell's value, so the network is not one city-wide number
    assert len(set(dry[0, :, precip].tolist())) == len(city["cells"]) > 1
    assert (wet[0, :, precip] > dry[0, :, precip]).all()
    # scaled on the training days: the driest cell-day is 0, the wettest 1
    both = torch.cat([dry[0, :, precip], wet[0, :, precip]])
    assert both.min() == pytest.approx(0.0) and both.max() == pytest.approx(1.0)


def test_a_day_without_weather_is_mid_range_and_reported(city):
    ctx = build(city)
    window = ctx.window(date(2030, 1, 1), "AM", 0)
    assert torch.allclose(window[..., COLUMN["precip"]], torch.tensor(0.5))
    assert date(2030, 1, 1) in ctx.missing_weather


def test_flood_is_ordinal_and_interacts_with_rain(city):
    ctx = build(city)
    w = ctx.window(DAY + timedelta(days=1), "AM", 0)[0]
    flood = torch.tensor(FLOOD, dtype=torch.float32) / 3
    assert torch.allclose(w[:, COLUMN["flood_level"]], flood)
    assert torch.allclose(w[:, COLUMN["flood_x_precip"]], flood * w[:, COLUMN["precip"]])


def test_incidents_decay_over_their_ttl_at_their_node(city):
    start = datetime(2026, 5, 4, 7, 0)
    nodes = event_node_index(city["graph"], city["dir"], None)
    ctx = build(city, events=[event(start + timedelta(minutes=10))], event_nodes=nodes)
    w = ctx.window(DAY, "AM", 0)                        # 7:00-7:29
    at = CAMERAS["C2"]
    impact = w[:, at, COLUMN["event_impact"]]
    assert impact[:10].sum() == 0                       # not yet reported
    assert impact[10] == pytest.approx(1.0)             # fresh two-lane incident
    assert impact[29] == pytest.approx(1.0 - 19 / EVENT_TTL_MINUTES)
    assert w[10, at, COLUMN["event_count"]] == pytest.approx(math.log1p(1))
    others = torch.ones(w.shape[1], dtype=torch.bool); others[at] = False
    assert w[:, others, COLUMN["event_impact"]].sum() == 0
    later = ctx.window(DAY, "AM", 75)                   # 8:15, past the 60-minute TTL
    assert later[:, at, COLUMN["event_impact"]].sum() == 0


def test_clock_is_the_real_minute_of_day_and_weekday(city):
    w = build(city).window(DAY, "PM", 30)               # starts 17:30 on a Monday
    minute = 17 * 60 + 30
    assert w[0, 0, COLUMN["minute_sin"]] == pytest.approx(math.sin(2 * math.pi * minute / 1440), abs=1e-6)
    assert w[0, 0, COLUMN["minute_cos"]] == pytest.approx(math.cos(2 * math.pi * minute / 1440), abs=1e-6)
    assert w[0, 0, COLUMN["weekday_sin"]] == pytest.approx(0.0, abs=1e-6)   # Monday = 0


def test_road_features_describe_the_network(city):
    w = build(city).window(DAY, "AM", 0)[0]
    cams = list(CAMERAS.values())
    assert (w[cams, COLUMN["camera_hops"]] == 0).all() and (w[cams, COLUMN["is_camera"]] == 1).all()
    assert w[:, COLUMN["camera_hops"]].max() > 0
    avenue = [AVENUE_ROW * SIDE + c for c in range(SIDE)]
    side_street = [5 * SIDE + c for c in range(1, SIDE - 1)]
    assert w[avenue, COLUMN["max_speed"]].min() > w[side_street, COLUMN["max_speed"]].max()
    corner, middle = 0, 2 * SIDE + 2
    assert w[corner, COLUMN["out_degree"]] < w[middle, COLUMN["out_degree"]]


def test_dropping_a_source_zeroes_exactly_its_columns(city):
    full, dropped = build(city).window(DAY, "AM", 0), build(city, drop=("weather",)).window(DAY, "AM", 0)
    cols = group_columns(["weather"])
    assert (dropped[..., cols] == 0).all()
    rest = [i for i in range(N_FEATURES) if i not in cols]
    assert torch.equal(full[..., rest], dropped[..., rest])
    with pytest.raises(ValueError):
        group_columns(["rainbow"])


def test_event_intersections_snap_only_when_close(city, tmp_path):
    lat, lon = coords(7)
    csv = tmp_path / "ev.csv"
    csv.write_text("intersection,lat,lon,source\nNear,%f,%f,x\nFar,15.5,122.0,x\n" % (lat + 0.001, lon))
    index = event_node_index(city["graph"], city["dir"], csv)
    assert index["Near"] == 7 and "Far" not in index and index["C2"] == CAMERAS["C2"]


def test_edge_features_follow_edge_index(city):
    feats = edge_features(city["graph"], city["dir"])
    assert feats.shape == (city["graph"].edge_index.shape[1], 2)
    src, dst = city["graph"].edge_index
    on_avenue = (src // SIDE == AVENUE_ROW) & (dst // SIDE == AVENUE_ROW)
    assert feats[on_avenue, 1].min() == pytest.approx(0.6) and feats[~on_avenue, 1].max() == pytest.approx(0.2)


# ---------------------------------------------------------------- the model
def tiny_model(city, context=True):
    torch.manual_seed(0)
    fusion = CNNLSTMFusion(text_dim=8, temporal_dim=5, image_size=32, patch_size=16, patch_embed_dim=16,
                           lstm_hidden_dim=32, visual_feature_dim=16, text_projection_dim=8)
    stgnn = RADRSTGNN(in_features=32, gcn_hidden=32, gcn_out=16, gru_hidden=32)
    if context:
        attr = edge_features(city["graph"], city["dir"])
        return TrafficRiskModel(fusion, stgnn, MLPDecoder(32, edge_feature_dim=2),
                                context_dim=N_FEATURES, edge_features=attr)
    return TrafficRiskModel(fusion, stgnn, MLPDecoder(32))


def make_batch(city, ctx_tensor=None, brightness=0.5, b=1, t=T):
    n_cam = len(CAMERAS)
    batch = {
        "images": torch.full((b, n_cam, t, 3, 32, 32), brightness),
        "text": torch.zeros(b, n_cam, t, 8),
        "temporal": torch.full((b, n_cam, t, 5), 0.5),
        "visual_mask": torch.ones(b, n_cam, t, dtype=torch.bool),
        "text_mask": torch.zeros(b, n_cam, t, dtype=torch.bool),
    }
    if ctx_tensor is not None:
        batch["context"] = ctx_tensor.expand(b, *ctx_tensor.shape).clone() if ctx_tensor.ndim == 3 else ctx_tensor
    return batch


def graph_args(city):
    g = city["graph"]
    cams = torch.tensor([g.camera_nodes[label] for label in sorted(CAMERAS)])
    return g.a_hat, cams, g.edge_index, camera_edge_ids(g.edge_index, cams, g.n_nodes)


def far_edges(city, min_hops=3):
    """Edges whose both ends are at least `min_hops` from every camera."""
    g = city["graph"]
    w = build(city).window(DAY, "AM", 0)[0]
    hops = w[:, COLUMN["camera_hops"]] * 8
    src, dst = g.edge_index
    return ((hops[src] >= min_hops) & (hops[dst] >= min_hops)).nonzero(as_tuple=True)[0]


def test_every_source_reaches_roads_far_from_any_camera(city):
    a_hat, cams, edge_index, cam_edges = graph_args(city)
    model = tiny_model(city).eval()
    batch = make_batch(city, build(city).window(DAY + timedelta(days=1), "AM", 0))
    far = far_edges(city)
    assert len(far) > 0
    base = torch.sigmoid(model(batch, a_hat, cams, edge_index))
    # Untrained, so the size of the effect means nothing -- only that there is a path at all
    # (the CCTV-only model below leaves these roads bit-for-bit unchanged).
    for source in ("weather", "flood", "events", "temporal", "spatial"):
        moved = (torch.sigmoid(model(perturb(batch, source), a_hat, cams, edge_index)) - base).abs()[0, far]
        assert moved.max() > 1e-6, f"{source} does not reach roads 3+ hops from a camera"


def test_the_cctv_only_model_cannot_reach_far_roads(city):
    """The defect being fixed: without the node context, nothing changes a road 3+ hops away."""
    a_hat, cams, edge_index, _ = graph_args(city)
    model = tiny_model(city, context=False).eval()
    far = far_edges(city)
    dark, bright = make_batch(city, brightness=0.1), make_batch(city, brightness=0.9)
    wet = make_batch(city); wet["temporal"][..., :3] = 1.0
    base = torch.sigmoid(model(dark, a_hat, cams, edge_index))[0, far]
    for other in (bright, wet):
        assert torch.allclose(torch.sigmoid(model(other, a_hat, cams, edge_index))[0, far], base)


def test_a_context_model_needs_a_context(city):
    a_hat, cams, edge_index, _ = graph_args(city)
    with pytest.raises(ValueError, match="context"):
        tiny_model(city)(make_batch(city), a_hat, cams, edge_index)


def test_trained_on_camera_edges_the_model_moves_risk_off_camera_for_every_source(city):
    """
    Train only on camera edges (as the real model is), with a synthetic congestion rule that
    depends on every source, then check each source moves the risk of roads with no camera --
    the thesis claim that the CRS is multimodal across the network, not a function of CCTV.
    """
    a_hat, cams, edge_index, cam_edges = graph_args(city)
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    model = tiny_model(city)
    ctx_builder = build(city)
    static = ctx_builder.window(DAY, "AM", 0)[0]              # spatial + flood columns
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    src, dst = edge_index

    steps = 6  # short windows keep this test quick; the effect does not depend on length

    def sample(b=8):
        ctx = static.expand(b, steps, *static.shape).clone()
        rain = torch.rand(b, 1, 1, generator=gen)
        ctx[..., COLUMN["precip"]] = rain
        ctx[..., COLUMN["flood_x_precip"]] = ctx[..., COLUMN["flood_level"]] * rain
        hour = torch.rand(b, 1, 1, generator=gen) * 24
        ctx[..., COLUMN["minute_sin"]] = torch.sin(2 * math.pi * hour / 24)
        ctx[..., COLUMN["minute_cos"]] = torch.cos(2 * math.pi * hour / 24)
        incident = (torch.rand(b, 1, ctx.shape[2], generator=gen) < 0.2).float()
        ctx[..., COLUMN["event_impact"]] = incident
        ctx[..., COLUMN["event_count"]] = incident * math.log1p(1)
        bright = torch.rand(b, generator=gen)
        batch = make_batch(city, ctx, b=b, t=steps)
        batch["images"] = bright.view(b, 1, 1, 1, 1, 1).expand_as(batch["images"]).clone()
        # node-level congestion: CCTV + rain on flood-prone roads + incidents + evening peak + slow roads
        node = (1.5 * (bright.view(b, 1) - 0.5) + 2.0 * ctx[:, 0, :, COLUMN["flood_x_precip"]]
                + 1.5 * ctx[:, 0, :, COLUMN["event_impact"]] - 1.0 * ctx[:, 0, :, COLUMN["minute_cos"]]
                - 1.5 * ctx[:, 0, :, COLUMN["max_speed"]])
        target = torch.sigmoid(node[:, src[cam_edges]] + node[:, dst[cam_edges]] - 1.0)
        return batch, target

    for _ in range(250):
        batch, target = sample()
        loss, _ = edge_risk_loss(model(batch, a_hat, cams, edge_index), cam_edges, target)
        opt.zero_grad(); loss.backward(); opt.step()

    batch, _ = sample(b=16)
    result = source_sensitivity(model, batch, a_hat, cams, edge_index, cam_edges)
    for source in SOURCES:
        assert result[source]["other_mean"] > 0.005, f"{source} barely moves non-camera roads: {result[source]}"
        assert result[source]["other_moved"] > 0.5, f"{source} moves too few non-camera roads: {result[source]}"

    # and outside the cameras the score is no longer a constant across windows
    with torch.no_grad():
        risk = torch.cat([torch.sigmoid(model(sample(b=8)[0], a_hat, cams, edge_index)) for _ in range(3)])
    assert constant_edges(risk, cam_edges) == 0.0
