"""
Road name per edge of the routing graph, so routes can say which roads they take.

Reads the OSM graph spatial_topology.py saved (data/raw/spatial/metro_manila_road_graph.graphml)
and writes data/processed/spatial/edge_names.csv: source_node_id, target_node_id, name
for every named edge. Streamed rather than loaded with osmnx -- the graphml is ~90MB,
and only three attributes are needed.

Parallel edges between the same two nodes keep the LAST one's name, matching
spatial_topology.build_adjacency, whose nx.DiGraph(G) collapse keeps the last edge's
data. An edge with several OSM names ("['Kamuning Road', 'Kamias Road']") gets the
first.

    python scripts/build_edge_names.py
"""

import argparse
import ast
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
NS = "{http://graphml.graphdrawing.org/xmlns}"


def clean_name(raw: Optional[str]) -> Optional[str]:
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    if raw.startswith("["):
        try:
            names = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return raw
        return str(names[0]).strip() if names else None
    return raw


def edge_names(graphml: Path) -> Dict[Tuple[int, int], Optional[str]]:
    name_key = None
    names: Dict[Tuple[int, int], Optional[str]] = {}
    for _, elem in ET.iterparse(graphml, events=("end",)):
        if elem.tag == f"{NS}key" and elem.get("for") == "edge" and elem.get("attr.name") == "name":
            name_key = elem.get("id")
        elif elem.tag == f"{NS}edge":
            name = None
            for data in elem.iter(f"{NS}data"):
                if data.get("key") == name_key:
                    name = clean_name(data.text)
            # Assigned even when unnamed, so a later unnamed parallel edge overrides an
            # earlier named one -- same as the adjacency's last-edge-wins collapse.
            names[(int(elem.get("source")), int(elem.get("target")))] = name
            elem.clear()
    return names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graphml", type=Path, default=REPO_ROOT / "data/raw/spatial/metro_manila_road_graph.graphml")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data/processed/spatial/edge_names.csv")
    a = p.parse_args()

    names = edge_names(a.graphml)
    named = sorted((u, v, n) for (u, v), n in names.items() if n)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_node_id", "target_node_id", "name"])
        writer.writerows(named)
    print(f"{len(named):,} of {len(names):,} edges named -> {a.out}")


if __name__ == "__main__":
    main()
