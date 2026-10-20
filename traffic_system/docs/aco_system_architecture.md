# ANTROUTE: ACO System Architecture

Download the architecture diagram: [PNG](aco_system_architecture.png) | [JPG](aco_system_architecture.jpg). Regenerate them with `python -m scripts.render_architecture` from `traffic_system`.

Scope: San Juan, Manila, Mandaluyong, Makati, and Pasig. Period: **June 1, 2025–May 31, 2026**, hourly. The user permits documented estimates. The release is a historical-period routing model, not a fully observed traffic archive.

```mermaid
flowchart TD
    subgraph S1["DATA PREPARATION"]
        T["Temporal Data<br/>Historical Weather and Traffic Estimates"] --> V["Data Validation and Preprocessing"]
        P["Spatial Data<br/>Road Network and Flood-Hazard Data"] --> V
    end
    subgraph S2["USER ROUTE REQUEST"]
        R["User Route Request<br/>Origin, Destination(s), Date and Time"] --> C["Condition Snapshot<br/>Traffic, Weather, Flood Risk and Road Closures"]
    end
    subgraph S3["ROUTING FACTOR WEIGHTING AND COST COMPUTATION"]
        F["Ten Routing Factors"] --> N["Factor Normalization<br/>Each factor scaled to 0–1"]
        N --> W["Weight Calculation (CRITIC Method)"]
        W --> E["Weighted Edge Cost Calculation<br/>Road cost = Σ (CRITIC weight × normalized factor)"]
    end
    subgraph S4["ROUTE OPTIMIZATION"]
        A["Ant Colony Optimization<br/>Heuristic guidance, pheromone updates, convergence-based stopping"]
    end
    subgraph S5["DYNAMIC ROUTE OUTPUT"]
        O["Recommended Optimal Route<br/>Route, distance and path cost"] --> D["Dynamic Condition Change Detection<br/>Time of day, weather or road closure, checked at each intersection"]
    end
    subgraph S6["PERFORMANCE EVALUATION"]
        M["Distance Travel, Path Cost, Computational Time, Convergence Rate"]
    end
    V --> R
    C --> F
    E --> A
    A --> O
    D -- "Update Edge Cost" --> E
    O --> M
```

## Where each stage is implemented

| Stage | Box | Implementation |
|---|---|---|
| Data preparation | Temporal data | `data/processed/temporal/weatherstack_hourly.csv` (historical provider weather); `traffic_hourly.csv` (estimated traffic, one row per road segment per hour, from `scripts/preprocess_traffic.py`) |
| | Spatial data | OSM road graph, five city boundaries, Project NOAH flood polygons (`data/raw/`) |
| | Validation and preprocessing | `scripts/preprocess_road_features.py`, `scripts/preprocess_traffic.py`, `scripts/build_study_network.py`, `scripts/audit_historical_inputs.py`; release integrity check in `app/study_policy.py` |
| User route request | Route request | Web planner (place search over road names and 4,006 OSM places, or map clicks); `POST /routes/plan`, `/routes/trip` |
| | Condition snapshot | `traffic_snapshot` (`historical_traffic.py`), `historical_snapshot` (`historical_weather.py`), mapped flood risk, manual closures (`condition_update.py`) |
| Factor weighting and cost | Ten factors, normalization, edge cost | `segment_values` / `prepare_graph` in `dynamic_weight.py` |
| | CRITIC weights | `critic.py`; computed at build time within each of 600 sampled hours (alternatives: all segments at that hour), then averaged (`apply_critic` in `build_study_network.py`) |
| Route optimization | ACO | `ant_colony_route` in `aco_routing.py` |
| Dynamic route output | Route and change detection | `simulate_trip` in `dynamic_trip.py`: on arrival at each node, rebuild the snapshot for the trip clock, weather and closures; when a road's cost changes by ≥10% or a road closes/reopens, recompute edge costs and re-run ACO from the current node |
| Performance evaluation | Four measures | `metrics.py`, `TripResult.evaluation`; `scripts/evaluate_sop.py` for the SOP experiments |

## Inputs and methods

Distance and intersections are calculated from mapped geometry/topology. Road classifications and signals come from OSM, with missing signals assumed absent and labeled. The September 2026 map is a static temporal surrogate. Flood risk is mapped NOAH hazard exposure, not observed hourly flooding.

Traffic congestion and travel time use the published BPR functional form with explicitly assumed hourly demand and weekend/road-type factors. Capacity uses a reference saturation flow and assumed signal green ratio; lane fallbacks are disclosed. Historical provider weather is mapped to city grids; rain-speed effects use transferred FHWA reference ranges. The time-of-day feature is how busy each road's type typically is at the selected hour (assumed hourly profile × road-type factor); the weather feature scales rain cost with each road's flood hazard.

Every feature's source, classification, formula, and assumptions are documented in [estimation_methods.md](estimation_methods.md), the configuration, generated registry, and app. Feature weights come from CRITIC computed within each sampled hour and averaged over a year-wide sample; normalization references are provisional objective settings. No TomTom or HERE dependency is introduced.

## Algorithms

| | Proposed: ANTROUTE | Baseline: Cheng (2023) |
|---|---|---|
| Module | `aco_routing.py` | `baseline_cheng.py` |
| Factors | ten features | path length, travel time, traffic flow (estimated) |
| Weights | CRITIC | AHP from the paper (0.637, 0.258, 0.105) |
| Search | ACO, destination-distance heuristic, convergence rule | eq. 8 heuristic, q0 transfer rule, local/global updates, stops on the study's convergence rule within NC_max (adapted: Cheng's Step 7 runs all NC_max iterations) |
| On a change during a trip | when a road's cost changed by ≥10% or a road closed/reopened: re-search from the current node with fresh pheromone | eq. 9 as printed on roads whose travel time changed, full re-search with the rescaled pheromone |

`/routes/plan` and `/routes/baseline` route one date/hour snapshot. `/routes/trip` (`dynamic_trip.py`) simulates a trip: it drives the planned route, checks conditions on arrival at each node (the hour follows the trip clock; scheduled events close or reopen roads or change the weather), and applies the algorithm's dynamic rule. Every driven road is scored with the ten-feature cost and travel time of the moment it is entered. A trip reports initial routing time (graph build and search at departure), dynamic re-routing time (graph rebuild, change handling and re-search per re-route), computational time (both plus condition checks that did not lead to a re-route), and convergence rate as N_conv (the iteration at which a search's best path cost had not improved for 20 consecutive iterations; a search that reaches the iteration limit first is not converged). `scripts/evaluate_sop.py` runs both algorithms on the same trips, events and seeds and answers SOP 1–3.

A pheromone-carry-over variant (CSPA: scaling the pheromone of changed roads by old/new cost and keeping the rest) was evaluated and removed: on the same trips it gave higher path cost than re-planning with fresh pheromone in every scenario, because the retained pheromone kept the colony near the pre-change route. Adaptations of the baseline to this network are listed in `baseline_cheng.py`; the results reproduce Cheng's mechanisms on this network, not the paper's original experiments.

## Execution

1. Verify the release artifacts and authorized period/mode.
2. Validate endpoints and the selected date/hour.
3. Load provider weather for that timestamp, or apply the explicitly selected weather scenario.
4. Read the modeled traffic of every road segment for that date and hour. Exclude manually closed segments.
5. Normalize the ten feature contributions and form segment costs.
6. Run ACO using pheromones, cost and destination-distance heuristics, cycle avoidance, and bounded backtracking. No separate shortest-path solver supplies the route.
7. Display the best feasible route found, four evaluations, and input provenance. Changed inputs clear the old result until recalculation.

Routes remain within the five-city union. Each run uses one consistent snapshot. Historical weather records vary by date; modeled traffic is not represented as independent observed hourly samples. Static road/flood features are held constant by assumption.

## Evaluation

Distance Travel sums segment lengths. Path Cost sums the configured feature costs. Computational Time covers graph preparation and ACO, excluding input acquisition and rendering. Convergence Rule (Aadil et al., 2016), applied to both algorithms: the search stops once the best path cost has not improved for 20 consecutive iterations after a feasible route exists, and N_conv is that iteration. A search that reaches the iteration limit first is recorded as not converged. Convergence does not prove global optimality.

Functional tests and a demand-assumption sensitivity report verify behavior, not the empirical accuracy of the estimated historical conditions.
