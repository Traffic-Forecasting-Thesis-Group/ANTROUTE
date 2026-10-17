"""Render the implemented architecture in a monochrome academic flowchart style."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "docs"


def render():
    fig, ax = plt.subplots(figsize=(12, 15))
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 100), ylim=(0, 106))
    ax.axis("off")
    fig.subplots_adjust(left=.025, right=.975, bottom=.015, top=.985)

    def text(x, y, value, size=10, bold=False):
        ax.text(x, y, value, ha="center", va="center", fontsize=size,
                fontfamily="DejaVu Sans", weight="bold" if bold else "normal",
                linespacing=1.4, color="#111111")

    def box(x, y, w, h, title, detail="", size=10):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                     boxstyle="round,pad=0.2,rounding_size=.8",
                     facecolor="white", edgecolor="#555555", linewidth=1.0, zorder=2))
        if detail:
            text(x, y+h*.22, title, size, True)
            text(x, y-h*.18, detail, size-1)
        else:
            text(x, y, title, size, True)

    def arrow(points, dashed=False):
        # Use explicit bends to keep connectors away from unrelated boxes.
        for start, end in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color="#444444", linewidth=.95,
                    linestyle="--" if dashed else "-", zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                     mutation_scale=11, linewidth=.95, color="#444444",
                     linestyle="--" if dashed else "-", zorder=1,
                     shrinkA=0, shrinkB=2))

    text(50, 103.4, "System Architecture of the Proposed ANTROUTE Model", 16)
    text(50, 100.6, "San Juan · Manila · Mandaluyong · Makati · Pasig", 10)

    box(17, 94, 28, 7, "Spatial Data", "OSM road network and boundaries\nProject NOAH flood-hazard polygons")
    box(50, 94, 28, 7, "Temporal Data", "Weatherstack historical hourly records\nJune 2025 – May 2026")
    box(83, 94, 28, 7, "Estimation Configuration", "Published reference models\nDocumented assumptions and settings")

    box(17, 82, 28, 8, "Graph and Feature Preparation", "Five-city filtering; directed topology\nDistance, intersections, signals, road type\nFlood-hazard exposure")
    box(50, 82, 28, 8, "Hourly Weather Preparation", "Validate 8,760 hourly timestamps\nAssign weather grids to cities\nPreserve provider-source provenance")
    box(83, 82, 28, 8, "Estimation Rules", "BPR congestion and travel time\nLane / capacity and peak-hour rules\nRain effects; feature weights")
    for x in (17, 50, 83):
        arrow([(x, 90.3), (x, 86.2)])
        arrow([(x, 77.8), (x, 74.2)])

    box(50, 71, 94, 6, "Prepared Study Release", "Road graph + hourly weather + feature provenance registry + integrity manifest")
    arrow([(50, 67.8), (50, 63.2)])

    box(17, 59, 28, 8, "User Route Request", "React web interface\nOrigin, destination, departure date / hour\nPlace search and map selection")
    box(50, 59, 28, 8, "FastAPI Validation", "Verify release and study period\nValidate endpoints and conditions\nPOST /routes/plan")
    box(83, 59, 28, 8, "Dynamic Conditions", "Time of Day\nRoad Closure: manual scenario\nWeather Conditions: historical / scenario")
    arrow([(31.2, 59), (35.8, 59)])
    arrow([(50, 54.8), (50, 51.2)])
    arrow([(83, 54.8), (83, 51.2)])

    box(50, 48, 94, 6, "Date / Hour Condition Snapshot", "Resolve modeled traffic and weather effects; exclude closed road segments")
    arrow([(50, 44.8), (50, 42.2)])
    box(50, 37.5, 94, 9, "Ten Input Features", "Distance · Traffic Congestion · Travel Time · Number of Intersections · Traffic Signal\nRoad Type · Time of Day · Weather · Flood Risk · Road Capacity", 11)

    arrow([(50, 32.8), (50, 30.2)])
    box(50, 27, 48, 6, "Dynamic Weighted Graph", "Normalize features and calculate\nweighted segment costs")
    arrow([(50, 23.8), (50, 21.7)])

    box(50, 17, 48, 9, "Ant Colony Optimization (ACO)", "Construct feasible routes; update pheromones\nCost and destination-distance heuristics\nStop by convergence rule or iteration limit")
    box(87, 17, 22, 10, "Evaluation", "Distance Travel\nPath Cost\nComputational Time\nConvergence Rule", 9.5)
    arrow([(74.2, 17), (75.8, 17)])
    arrow([(50, 12.3), (50, 10.2)])
    box(50, 6.5, 62, 7, "Best Feasible Route Found", "Web map, route details and convergence history\nEvaluation results and disclosed feature sources / estimates")
    arrow([(87, 11.8), (87, 6.5), (81.2, 6.5)])

    text(50, .5, "June 1, 2025 – May 31, 2026 · Hourly, Asia/Manila · Historical routing with documented estimates", 9)
    text(50, -1.2, "2026 road map and static flood exposure are temporal proxies; historical traffic is modeled. ACO does not guarantee global optimality.", 8)
    png = OUT / "aco_system_architecture.png"
    jpg = OUT / "aco_system_architecture.jpg"
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    with Image.open(png) as img:
        img.convert("RGB").save(jpg, quality=95, subsampling=0, dpi=(300, 300))
        print(f"Exported PNG and JPG: {img.width} x {img.height}")


if __name__ == "__main__":
    render()
