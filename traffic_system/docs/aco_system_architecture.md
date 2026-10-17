# ANTROUTE: ACO System Architecture

Download the architecture diagram: [PNG](aco_system_architecture.png) | [JPG](aco_system_architecture.jpg). Both exports use a monochrome academic flowchart layout at 3600 × 4500 pixels and reflect the implemented codebase. Regenerate them with `python scripts/render_architecture.py` from `traffic_system`.

Scope: San Juan, Manila, Mandaluyong, Makati, and Pasig. Period: **June 1, 2025–May 31, 2026**, hourly. The user permits documented estimates. The release is a historical-period routing model, not a fully observed traffic archive.

```mermaid
flowchart TD
    A["Mapped road inventory and flood hazard polygons"] --> C["Five-city graph preparation"]
    B["Historical hourly provider weather<br/>June 2025–May 2026"] --> D["Date/hour snapshot"]
    P["Documented estimation settings<br/>Formulas, assumptions, source references"] --> C
    P --> D
    C --> D
    R["Origin, destination, date and hour"] --> D
    X["Dynamics<br/>Time of day<br/>Manual closure scenarios<br/>Historical weather or weather scenarios"] --> D
    D --> F["Ten-feature costs<br/>Distance, congestion, travel time<br/>Intersections, signals, road type<br/>Time of day, weather, flood risk, capacity"]
    F --> G["Ant Colony Optimization<br/>Route construction, pheromone updates<br/>Convergence rule"]
    G --> H["Best feasible route found"]
    G --> E["Evaluation<br/>Distance Travel<br/>Path Cost<br/>Computational Time<br/>Convergence Rule"]
```

## Inputs and methods

Distance and intersections are calculated from mapped geometry/topology. Road classifications and signals come from OSM, with missing signals assumed absent and labeled. The September 2026 map is a static temporal surrogate. Flood risk is mapped NOAH hazard exposure, not observed hourly flooding.

Traffic congestion and travel time use the published BPR functional form with explicitly assumed hourly demand and weekend/road-type factors. Capacity uses a reference saturation flow and assumed signal green ratio; lane fallbacks are disclosed. Historical provider weather is mapped to city grids; rain-speed effects use transferred FHWA reference ranges. The time-of-day feature uses the exact timestamp hour with a separate assumed peak-hours cost policy.

Every feature's source, classification, formula, and assumptions are documented in [estimation_methods.md](estimation_methods.md), the configuration, generated registry, and app. Equal feature weights and normalization references are provisional objective settings. No TomTom or HERE dependency is introduced.

## Execution

1. Verify the release artifacts and authorized period/mode.
2. Validate endpoints and the selected date/hour.
3. Load provider weather for that timestamp, or apply the explicitly selected weather scenario.
4. Resolve modeled traffic from hour, weekday, and road type. Exclude manually closed segments.
5. Normalize the ten feature contributions and form segment costs.
6. Run ACO using pheromones, cost and destination-distance heuristics, cycle avoidance, and bounded backtracking. No separate shortest-path solver supplies the route.
7. Display the best feasible route found, four evaluations, and input provenance. Changed inputs clear the old result until recalculation.

Routes remain within the five-city union. Each run uses one consistent snapshot. Historical weather records vary by date; modeled traffic is not represented as independent observed hourly samples. Static road/flood features are held constant by assumption.

## Evaluation

Distance Travel sums segment lengths. Path Cost sums the configured feature costs. Computational Time covers graph preparation and ACO, excluding input acquisition and rendering. Convergence Rule requires relative best-cost improvement within tolerance for the configured consecutive iterations after a feasible route exists; iteration-limit stops are reported separately. Convergence does not prove global optimality.

Functional tests and a demand-assumption sensitivity report verify behavior, not the empirical accuracy of the estimated historical conditions.
