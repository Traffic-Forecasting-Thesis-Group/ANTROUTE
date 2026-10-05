from __future__ import annotations
import argparse
import glob
import io
import logging
import os
import zipfile
from pathlib import Path

os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
import scipy.sparse as sp
from shapely.geometry import Point

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("spatial_topology")
ox.settings.use_cache = True
ox.settings.log_console = False
PLACE = "Metro Manila, Philippines"
EDSA_ALIASES = ["edsa", "epifanio de los santos avenue", "epifanio delos santos avenue"]
ROXAS_ALIASES = ["roxas boulevard", "roxas blvd"]
INTERSECTIONS = {
    "EDSA-Quezon Ave": (EDSA_ALIASES, ["quezon avenue", "quezon ave"]),
    "EDSA-Kamuning": (EDSA_ALIASES, ["kamuning road", "kamuning rd"]),
    "EDSA-Aurora": (EDSA_ALIASES, ["aurora boulevard", "aurora blvd"]),
    "EDSA-Regalia (P. Tuazon)": (EDSA_ALIASES, ["p. tuazon", "p tuazon", "tuazon"]),
    "EDSA-Ortigas-Shaw": (EDSA_ALIASES, ["shaw boulevard", "shaw blvd", "ortigas avenue"]),
    "8337-Ayala NB 1-PTZ": (EDSA_ALIASES, ["ayala avenue", "ayala ave"]),
    "Roxas Blvd-Kalaw": (ROXAS_ALIASES, ["kalaw avenue", "kalaw street", "kalaw"]),
    "Roxas-Padre Burgos": (ROXAS_ALIASES, ["padre burgos", "p. burgos", "p burgos"]),
}
# Roughly where each camera intersection really is (lat, lon). Street names alone are ambiguous:
# OSM has two places where an "EDSA" road meets an "Aurora Boulevard" -- the camera's one in Cubao,
# and Aurora Blvd (Tramo) in Pasay, 9 km south -- and picking the first name match put the
# EDSA-Aurora camera in Pasay. A name match only counts within ANCHOR_RADIUS_M of the anchor.
INTERSECTION_ANCHORS = {
    "EDSA-Quezon Ave": (14.6442, 121.0377),
    "EDSA-Kamuning": (14.6307, 121.0457),
    "EDSA-Aurora": (14.6216, 121.0501),        # Cubao, NOT Aurora Blvd (Tramo), Pasay
    "EDSA-Regalia (P. Tuazon)": (14.6165, 121.0523),
    "EDSA-Ortigas-Shaw": (14.5934, 121.0583),  # the Ortigas Ave crossing; Shaw is a separate one
    "8337-Ayala NB 1-PTZ": (14.5501, 121.0294),
    "Roxas Blvd-Kalaw": (14.5798, 120.9774),
    "Roxas-Padre Burgos": (14.5830, 120.9755),
}
ANCHOR_RADIUS_M = 500.0
MANUAL_OVERRIDES: dict[str, int] = {}
HF_REPO = "bettergovph/project-noah-hazard-maps"
HF_API_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"


def _street_names(data: dict) -> list[str]:
    name = data.get("name")
    if name is None:
        return []
    return [n.lower() for n in (name if isinstance(name, list) else [name])]


def _matches_any(street_names: list[str], aliases: list[str]) -> bool:
    return any((alias in sn or sn in alias for sn in street_names for alias in aliases))


def _to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() == "true"


def load_or_build_graph(graph_path: Path) -> nx.MultiDiGraph:
    if graph_path.exists():
        log.info(f"Found existing graph at {graph_path}. Loading instead of re-downloading.")
        return ox.load_graphml(graph_path)
    log.info(f"Downloading drivable road network for: {PLACE} ...")
    return ox.graph_from_place(PLACE, network_type="drive", simplify=True)


def _distance_m(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6_371_000.0 * np.arcsin(np.sqrt(h)))


def locate_key_intersections(G: nx.MultiDiGraph) -> dict[str, int]:
    # A graph loaded from a previous run's graphml still carries that run's CCTV flags; clear them,
    # or a node this run no longer picks would stay marked as a camera next to the new one.
    for _, data in G.nodes(data=True):
        data.pop("is_cctv_node", None)
        data.pop("cctv_label", None)
    node_streets: dict[int, set[str]] = {}
    for u, v, data in G.edges(data=True):
        names = _street_names(data)
        if not names:
            continue
        node_streets.setdefault(u, set()).update(names)
        node_streets.setdefault(v, set()).update(names)
    key_nodes: dict[str, int] = {}
    for label, (aliases_a, aliases_b) in INTERSECTIONS.items():
        if label in MANUAL_OVERRIDES:
            key_nodes[label] = MANUAL_OVERRIDES[label]
            continue
        candidates = [
            node
            for node, streets in node_streets.items()
            if _matches_any(streets, aliases_a) and _matches_any(streets, aliases_b)
        ]
        if label in INTERSECTION_ANCHORS:
            lat, lon = INTERSECTION_ANCHORS[label]
            far = [n for n in candidates
                   if _distance_m(lat, lon, G.nodes[n]["y"], G.nodes[n]["x"]) > ANCHOR_RADIUS_M]
            if far:
                log.info(f"'{label}': ignoring {len(far)} same-name match(es) more than "
                         f"{ANCHOR_RADIUS_M:.0f} m from the intersection, e.g. node {far[0]}")
            candidates = [n for n in candidates if n not in far]
        if not candidates:
            log.warning(f"No graph match for '{label}'")
            continue
        key_nodes[label] = candidates[0]
        if len(candidates) > 1:
            log.info(f"'{label}' matched {len(candidates)} candidates - using the first")
    for name, node_id in key_nodes.items():
        G.nodes[node_id]["is_cctv_node"] = True
        G.nodes[node_id]["cctv_label"] = name
    log.info(f"Located {len(key_nodes)}/{len(INTERSECTIONS)} key intersections")
    for name, node_id in key_nodes.items():
        log.info(f"  {name:30s} -> node {node_id}")
    return key_nodes


HWY_SPEEDS_KPH = {
    "motorway": 80,
    "motorway_link": 60,
    "trunk": 60,
    "trunk_link": 45,
    "primary": 50,
    "primary_link": 40,
    "secondary": 40,
    "secondary_link": 30,
    "tertiary": 30,
    "tertiary_link": 25,
    "residential": 20,
    "living_street": 10,
    "unclassified": 30,
    "service": 15,
}
FALLBACK_SPEED_KPH = 30
HWY_LANES_FALLBACK = {
    "motorway": 3,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 1,
    "primary": 2,
    "primary_link": 1,
    "secondary": 2,
    "secondary_link": 1,
    "tertiary": 1,
    "tertiary_link": 1,
    "residential": 1,
    "living_street": 1,
    "unclassified": 1,
    "service": 1,
}
FALLBACK_LANES = 1


def add_free_flow_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    G = ox.routing.add_edge_speeds(G, hwy_speeds=HWY_SPEEDS_KPH, fallback=FALLBACK_SPEED_KPH)
    G = ox.routing.add_edge_travel_times(G)
    return G


def _first_value(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _road_type(data: dict) -> str:
    value = _first_value(data.get("highway"))
    return str(value) if value is not None else "unclassified"


def _lane_count(data: dict) -> int:
    raw = _first_value(data.get("lanes"))
    if raw is not None:
        try:
            parsed = int(float(raw))
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass
    road_type = _road_type(data)
    return HWY_LANES_FALLBACK.get(road_type, FALLBACK_LANES)


def add_lane_counts(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    imputed = 0
    for _, _, data in G.edges(data=True):
        raw = _first_value(data.get("lanes"))
        count = _lane_count(data)
        data["lanes"] = count
        if raw is None:
            imputed += 1
    log.info(
        f"Lane counts: {imputed:,} of {G.number_of_edges():,} edges had no OSM 'lanes' tag and were filled in by road type (fallback {FALLBACK_LANES})."
    )
    return G


def build_travel_time_matrix(G: nx.MultiDiGraph, node_order):
    G_simple = nx.DiGraph(G)
    return nx.adjacency_matrix(G_simple, nodelist=node_order, weight="travel_time")


def build_speed_matrix(G: nx.MultiDiGraph, node_order):
    G_simple = nx.DiGraph(G)
    return nx.adjacency_matrix(G_simple, nodelist=node_order, weight="speed_kph")


def build_lane_matrix(G: nx.MultiDiGraph, node_order):
    G_simple = nx.DiGraph(G)
    return nx.adjacency_matrix(G_simple, nodelist=node_order, weight="lanes")


def build_edge_features_table(G: nx.MultiDiGraph, node_order) -> pd.DataFrame:
    G_simple = nx.DiGraph(G)
    rows = []
    for u, v, data in G_simple.edges(data=True):
        rows.append(
            {
                "source_node_id": u,
                "target_node_id": v,
                "length_m": data.get("length"),
                "road_type": _road_type(data),
                "lanes": data.get("lanes"),
                "speed_kph": data.get("speed_kph"),
                "travel_time_s": data.get("travel_time"),
            }
        )
    return pd.DataFrame(rows)


def build_adjacency(G: nx.MultiDiGraph):
    G_simple = nx.DiGraph(G)
    node_order = list(G_simple.nodes())
    adj_matrix = nx.adjacency_matrix(G_simple, nodelist=node_order, weight="length")
    log.info(
        f"Adjacency matrix: {adj_matrix.shape[0]:,} x {adj_matrix.shape[1]:,}, {adj_matrix.nnz:,} nonzero entries"
    )
    return (adj_matrix, node_order)


def fetch_flood_hazard(
    spatial_raw_dir: Path, hazard_folder: str = "Flood/5yr", name_hints=("manila", "ncr", "metro")
) -> gpd.GeoDataFrame:
    cache_path = spatial_raw_dir / "flood_hazard.geojson"
    if cache_path.exists():
        log.info(f"Found existing data at {cache_path}. Skipping download.")
        return gpd.read_file(cache_path)
    resp = requests.get(f"{HF_API_TREE}/{hazard_folder}", timeout=15)
    resp.raise_for_status()
    zip_path = None
    for entry in resp.json():
        path = entry.get("path", "")
        fname = path.rsplit("/", 1)[-1].lower()
        if fname.endswith(".zip") and any((hint in fname for hint in name_hints)):
            zip_path = path
            break
    if zip_path is None:
        raise FileNotFoundError(f"No file matching {name_hints} in '{hazard_folder}'")
    log.info(f"Found hazard zip: {zip_path}")
    resp = requests.get(f"{HF_RESOLVE}/{zip_path}", timeout=60)
    resp.raise_for_status()
    extract_dir = spatial_raw_dir / "noah_flood_metro_manila"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(extract_dir)
    shp_files = glob.glob(f"{extract_dir}/**/*.shp", recursive=True)
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found inside '{extract_dir}/'")
    flood_gdf = gpd.read_file(shp_files[0])
    if flood_gdf.crs is not None and flood_gdf.crs.to_epsg() != 4326:
        flood_gdf = flood_gdf.to_crs(epsg=4326)
    flood_gdf.to_file(cache_path, driver="GeoJSON")
    log.info(f"Saved fetched data to {cache_path}")
    return flood_gdf


def join_flood_hazard(full_nodes_df: pd.DataFrame, flood_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    nodes_gdf = gpd.GeoDataFrame(
        full_nodes_df,
        geometry=[Point(lon, lat) for lat, lon in zip(full_nodes_df["lat"], full_nodes_df["lon"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(nodes_gdf, flood_gdf[["Var", "geometry"]], how="left", predicate="within")
    joined["Var"] = joined["Var"].fillna(0).astype(int)
    flood_by_node = joined.groupby("node_id")["Var"].max().rename("flood_hazard_level")
    out = full_nodes_df.merge(flood_by_node, on="node_id", how="left")
    out["flood_hazard_level"] = out["flood_hazard_level"].fillna(0).astype(int)
    return out


def plot_topology(G, flood_gdf, key_nodes, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 12))
    hazard_colors = {1: "#FFF3B0", 2: "#FFA94D", 3: "#E03131"}
    for level, color in hazard_colors.items():
        subset = flood_gdf[flood_gdf["Var"] == level]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, alpha=0.5, edgecolor="none", label=f"Hazard level {level}")
    ox.plot_graph(
        G,
        ax=ax,
        node_size=0,
        edge_color="#606060",
        edge_linewidth=0.4,
        bgcolor="white",
        show=False,
        close=False,
    )
    for name, node_id in key_nodes.items():
        y, x = (G.nodes[node_id]["y"], G.nodes[node_id]["x"])
        ax.scatter(x, y, c="gold", s=80, zorder=5, edgecolors="black", linewidths=0.8, marker="*")
        txt = ax.annotate(name, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points", color="gold")
        txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground="black")])
    legend_elements = [
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#FFF3B0", markersize=10, label="Low hazard"
        ),
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#FFA94D", markersize=10, label="Medium hazard"
        ),
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#E03131", markersize=10, label="High hazard"
        ),
        plt.Line2D(
            [0], [0], marker="*", color="w", markerfacecolor="gold", markersize=12, label="CCTV intersection"
        ),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.set_title("Metro Manila Road Network with Flood Hazard Zones and CCTV Intersections", fontsize=10)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved topology plot to {out_path}")


def run(data_dir: Path, force_rebuild: bool = False, make_plot: bool = True) -> None:
    raw_dir = data_dir / "raw" / "spatial"
    proc_dir = data_dir / "processed" / "spatial"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    graph_path = raw_dir / "metro_manila_road_graph.graphml"
    if force_rebuild and graph_path.exists():
        graph_path.unlink()
    G = load_or_build_graph(graph_path)
    log.info(f"Full network: {len(G.nodes):,} nodes, {len(G.edges):,} edges")
    key_nodes = locate_key_intersections(G)
    adj_matrix, node_order = build_adjacency(G)
    sp.save_npz(proc_dir / "metro_manila_adjacency.npz", adj_matrix.tocsr())
    np.save(proc_dir / "metro_manila_node_order.npy", np.array(node_order))
    G = add_free_flow_travel_times(G)
    G = add_lane_counts(G)
    travel_time_matrix = build_travel_time_matrix(G, node_order)
    sp.save_npz(proc_dir / "metro_manila_travel_time.npz", travel_time_matrix.tocsr())
    log.info(
        f"Saved free-flow travel times: {travel_time_matrix.nnz:,} edges, median {np.median(travel_time_matrix.tocoo().data):.1f}s"
    )
    speed_matrix = build_speed_matrix(G, node_order)
    sp.save_npz(proc_dir / "metro_manila_speed_kph.npz", speed_matrix.tocsr())
    lane_matrix = build_lane_matrix(G, node_order)
    sp.save_npz(proc_dir / "metro_manila_lanes.npz", lane_matrix.tocsr())
    log.info(f"Saved speed limits (kph) and lane counts for {lane_matrix.nnz:,} edges")
    edge_features_df = build_edge_features_table(G, node_order)
    edge_features_df.to_csv(proc_dir / "full_network_edge_features.csv", index=False)
    log.info(
        f"Saved full_network_edge_features.csv: {len(edge_features_df):,} edges with road_type, lanes, speed_kph, travel_time_s, length_m"
    )
    ox.save_graphml(G, filepath=graph_path)
    key_node_ids = list(key_nodes.values())
    key_idx = [node_order.index(n) for n in key_node_ids if n in node_order]
    key_submatrix = adj_matrix[np.ix_(key_idx, key_idx)]
    sp.save_npz(proc_dir / "key_intersections_adjacency.npz", sp.csr_matrix(key_submatrix))
    key_order_rows = [
        {
            "row_col_index": i,
            "node_id": node_order[idx],
            "cctv_label": G.nodes[node_order[idx]].get("cctv_label"),
        }
        for i, idx in enumerate(key_idx)
    ]
    pd.DataFrame(key_order_rows).to_csv(proc_dir / "key_intersections_node_order.csv", index=False)
    log.info(
        "Saved: metro_manila_adjacency.npz, metro_manila_node_order.npy, metro_manila_road_graph.graphml, key_intersections_adjacency.npz, key_intersections_node_order.csv"
    )
    node_rows = [
        {
            "node_id": node_id,
            "lat": data["y"],
            "lon": data["x"],
            "is_cctv_node": _to_bool(data.get("is_cctv_node")),
            "cctv_label": data.get("cctv_label") or None,
        }
        for node_id, data in G.nodes(data=True)
    ]
    full_nodes_df = pd.DataFrame(node_rows)
    flood_gdf = fetch_flood_hazard(raw_dir)
    full_nodes_df = join_flood_hazard(full_nodes_df, flood_gdf)
    log.info(f"Flood hazard computed for all {len(full_nodes_df):,} nodes")
    log.info(full_nodes_df["flood_hazard_level"].value_counts().to_dict())
    out_csv = proc_dir / "full_network_static_features.csv"
    full_nodes_df.to_csv(out_csv, index=False)
    log.info(f"Saved: {out_csv}")
    if make_plot:
        plot_topology(G, flood_gdf, key_nodes, proc_dir / "metro_manila_topology_with_hazard.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Build Metro Manila spatial graph + flood hazard features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(data_dir=args.data_dir, force_rebuild=args.force_rebuild, make_plot=not args.no_plot)