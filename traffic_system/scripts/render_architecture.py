"""Render the ANTROUTE system architecture: six stages from data preparation to evaluation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "docs"
# (title, fill, edge, top, bottom) of each stage panel, in diagram units.
STAGES = [
    ("DATA PREPARATION", "#dae8fc", "#6c8ebf", 139, 111),
    ("USER ROUTE REQUEST", "#e8e8f8", "#6c6c9f", 107, 84),
    ("ROUTING FACTOR WEIGHTING AND COST COMPUTATION", "#d5e8d4", "#82b366", 80, 40),
    ("ROUTE OPTIMIZATION", "#ffe6cc", "#d79b00", 36, 24),
    ("DYNAMIC ROUTE OUTPUT", "#fff2cc", "#d6b656", 20, 6.5),
    ("PERFORMANCE EVALUATION", "#e1d5e7", "#9673a6", 2.5, -9),
]


def render():
    fig, ax = plt.subplots(figsize=(9, 18))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 100), ylim=(-10, 140))
    ax.axis("off")
    fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.99)

    def text(x, y, value, size=9, bold=False):
        ax.text(x, y, value, ha="center", va="center", fontsize=size, fontfamily="DejaVu Sans",
                weight="bold" if bold else "normal", linespacing=1.35, color="#111111", zorder=4)

    def box(x, y, w, h, title, detail=""):
        ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="white", edgecolor="#222222", linewidth=.9, zorder=2))
        if detail:
            lines = detail.count("\n") + 1
            text(x, y + 1.1 * lines / 2 + .3, title, 9, True)
            text(x, y - .9, detail, 8.3)
        else:
            text(x, y, title, 9, True)

    def arrow(points):
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]], color="#222222", linewidth=.9, zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=10,
                                     linewidth=.9, color="#222222", zorder=3, shrinkA=0, shrinkB=0))

    for title, fill, edge, top, bottom in STAGES:
        ax.add_patch(Rectangle((2, bottom), 81, top - bottom, facecolor=fill, edgecolor=edge, linewidth=1, zorder=0))
        text(42.5, top - 2, title, 9.5, True)

    box(22, 131, 34, 6, "Temporal Data", "Historical Weather and Traffic Estimates")
    box(63, 131, 34, 6, "Spatial Data", "Road Network and Flood-Hazard Data")
    box(42.5, 117.5, 75, 6, "Data Validation and Preprocessing", "Validates, cleans, and standardizes data for routing.")
    arrow([(22, 128), (22, 120.5)])
    arrow([(63, 128), (63, 120.5)])
    arrow([(42.5, 111), (42.5, 107)])

    box(42.5, 98.5, 75, 6, "User Route Request", "Origin, Destination(s), Date and Time")
    box(42.5, 88.5, 75, 6, "Condition Snapshot", "Traffic, Weather, Flood Risk and Road Closures")
    arrow([(42.5, 95.5), (42.5, 91.5)])
    arrow([(42.5, 84), (42.5, 80)])

    box(42.5, 72, 75, 7, "Ten Routing Factors", "Distance, congestion, travel time, intersections, signals, road type,\ntime of day, weather, flood risk, road capacity")
    box(42.5, 62, 75, 6, "Factor Normalization", "Each factor scaled to 0–1")
    box(42.5, 53, 75, 6, "Weight Calculation (CRITIC Method)", "Calculates weights for the ten routing factors.")
    box(42.5, 44, 75, 7, "Weighted Edge Cost Calculation", "Calculates each road segment's cost\nRoad cost = Σ (CRITIC weight × normalized factor)")
    arrow([(42.5, 68.5), (42.5, 65)])
    arrow([(42.5, 59), (42.5, 56)])
    arrow([(42.5, 50), (42.5, 47.5)])
    arrow([(42.5, 40), (42.5, 36)])

    box(42.5, 28.5, 75, 6, "Ant Colony Optimization", "Find optimal route using heuristic guidance, pheromone updates,\nand convergence-based stopping.")
    arrow([(42.5, 24), (42.5, 20)])

    box(21, 11.5, 34, 6, "Recommended Optimal Route", "Route, distance and path cost")
    box(63, 11.5, 34, 6, "Dynamic Condition Change Detection", "Time of day, weather or road closure,\nchecked at each intersection")
    arrow([(38, 11.5), (46, 11.5)])
    arrow([(80, 11.5), (92, 11.5), (92, 44), (80, 44)])
    ax.text(93, 28, "Update Edge Cost", rotation=90, ha="left", va="center", fontsize=8.5, fontfamily="DejaVu Sans", zorder=4)
    arrow([(21, 8.5), (21, 2.5)])

    box(42.5, -4, 75, 6, "Distance Travel, Path Cost, Computational Time, Convergence Rate")

    png = OUT / "aco_system_architecture.png"
    jpg = OUT / "aco_system_architecture.jpg"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    with Image.open(png) as img:
        img.convert("RGB").save(jpg, quality=95, subsampling=0, dpi=(300, 300))
        print(f"Exported PNG and JPG: {img.width} x {img.height}")


if __name__ == "__main__":
    render()
