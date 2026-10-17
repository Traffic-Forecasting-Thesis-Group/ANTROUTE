# Estimation methods and sources

Period: **June 1, 2025 through May 31, 2026**, hourly, Asia/Manila. Scope: San Juan, Manila, Mandaluyong, Makati, and Pasig. Estimates are explicitly permitted. This is a routing model with historical provider-weather inputs, not a reconstructed archive of measured traffic on every road.

The executable settings are in [study_assumptions.json](../configs/study_assumptions.json). Every feature is disclosed in `data/processed/estimation_registry.json`, the network provenance, route responses, and the app's **Where each feature and estimate comes from** table.

## Ten-feature basis

| Feature | Basis and computation | What is assumed or uncertain |
|---|---|---|
| Distance | WGS84 geodesic polyline length using PyProj. OSMnx single-segment endpoint geometry follows its conversion convention. | September 2026 mapped geometry is used as a static surrogate throughout the study year. It is not a surveyed historical length. |
| Traffic Congestion | BPR-derived slowdown index `1 - t0/t`, where `t/t0 = 1 + 0.15*(V/C)^4`. | All V/C values are assumed demand scenarios; there are no measured traffic volumes or historical congestion records. |
| Travel Time | `t0 = length / (speed_kph/3.6)`, followed by the BPR factor and the weather multiplier. | Posted OSM speed is a free-flow proxy. Missing speeds use the explicit road-type table. Neither is observed vehicle speed. |
| Number of Intersections | Count entered mapped nodes with street count at least three; exclude the origin. | Graph topology is held constant across the year and may omit physical junction details. |
| Traffic Signal | OSM `traffic_signals` node tags. | Untagged signals are assumed absent; absence is not verified. Historical installations/removals are unknown. |
| Road Type | OSM highway classifications. | Types are held constant. The categorical cost table is an objective preference, not a measured road property. |
| Time of Day | Actual selected hour and weekday from the request timestamp. A separate binary peak-hour policy marks 07, 08, 17, 18, and 19. | Peak hours and their objective contribution are project assumptions. This vector is not the congestion vector, though the effects are correlated. |
| Weather | Actual dated rows from the retained Weatherstack hourly CSV. Choose the nearest 0.05-degree grid to each city's representative point. | The city-grid assignment is a spatial approximation, and the provider's original measurement/model provenance is not independently verified. Speed effects are transferred reference estimates. |
| Flood Risk | Maximum intersected NOAH hazard class divided by three. | Static modeled hazard exposure, not an hourly flooding probability or observed depth. No overlap is not proof of safety; bridge elevation is not corrected. |
| Road Capacity | Directional lane estimate multiplied by saturation flow and effective green ratio. | Lane defaults, signal timing, traffic composition, and unsignalized capacity are unmeasured. This is a capacity proxy, not observed throughput. |

## Published functional forms

**BPR travel time.** The implementation uses `t = t0 * (1 + alpha*(V/C)^beta)`, with reference coefficients `alpha=0.15`, `beta=4`. This supplies a functional form, not evidence for our assumed demand. [FHWA reference](https://www.fhwa.dot.gov/Environment/air_quality/conformity/research/sample_methodologies/emismeth03.cfm).

**Signalized capacity.** The reference relationship is `c = s * g/C`. The model uses `s = 1900` passenger-car-equivalent vehicles/hour/lane, assumes `g/C=0.5` at a mapped downstream signal, and otherwise uses 1. It does not model detailed lane movements, vehicle mix, or signal plans. Applying this simplified proxy to unsignalized links is a project extrapolation. [Capacity relationship](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter3.htm) · [Saturation-flow reference](https://ops.fhwa.dot.gov/publications/fhwahop09008/chapter3.htm).

**Rain speed effects.** FHWA reports light-rain free-flow speed reductions of 2–13% and heavy-rain reductions of 6–17%. The model chooses the range midpoints, 7.5% and 11.5%, and converts speed reduction `r` to time multiplier `1/(1-r)`. These are transferred assumptions, not Philippine urban-road calibration. [FHWA road-weather reference](https://ops.fhwa.dot.gov/weather/roadimpact.htm).

Weather scenarios have categorical costs 0, 0.4, and 0.8. Historical precipitation/description chooses clear, rain, or heavy rain; the threshold of 7.5 mm in the provider precipitation field is a project classification rule. Neither categorical costs nor this threshold are claimed to come from the FHWA speed study. Weather does not invent flood closures.

## Project assumptions without empirical calibration

The configuration records all 24 hourly V/C values. Each is multiplied by a road-type factor; weekend requests multiply demand by 0.75. These numbers are scenario parameters chosen for the implementation, not reported Metro Manila traffic statistics. They must remain labeled as such.

Road-type fallback speeds, fallback lanes, categorical road costs, the assumed effective green ratio, the peak-hours list, and the equal ten-feature weights are also provisional choices. Directional lane tags are preferred; two-way total lane tags are split between directions; otherwise the configured fallback is used. Unknown signals are assumed absent and reported as unrecorded.

The objective normalizes distance against 1,000 m, travel time against 600 s, and capacity against 4,000 equivalent vehicles/hour; it caps normalized values at one. These references and 10% feature weights are software settings, not measured input statistics. Correlated inputs can overlap in the objective.

## Historical coverage and provenance

The source CSV has 34 grid cells, each with 8,760 hourly records. The builder verifies all expected timestamps and rejects duplicate or missing timestamps for the selected grids. It saves a source hash and the exact city-to-grid assignments in `data/processed/historical_weather.json`. Cities may share a grid; those are not independent city measurements. No live October 2026 weather is used for a study route.

Traffic varies by selected hour and weekday through the disclosed model. Historical weather varies by actual CSV timestamp. This does **not** create 8,760 independent observed traffic samples. Road geometry, classifications, signals, and mapped flood exposure are static surrogates.

Manually entered closures are counterfactual scenarios, not claims that those roads were actually closed on the selected date. Clear/rain/heavy-rain overrides are likewise labeled scenarios; the historical-weather option uses the dated provider records.

Mapped sources: [OSM attribution](https://www.openstreetmap.org/copyright), [OSMnx conversion convention](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.convert.graph_to_gdfs), [NOAH hazard methodology](https://noah.up.edu.ph/know-your-hazards). Weather comes from the retained `weatherstack_hourly.csv`; the provider's historical endpoint was separately verified to return hourly records, but API access does not establish measurement provenance.

## Verification and sensitivity

The release manifest hashes the network, weather bundle, assumptions, and period policy. Changed or missing artifacts block routing until the release is rebuilt. This checks artifact integrity, not scientific validity.

`python -m scripts.validate_system` checks real-network routing and closure exclusion. `python -m scripts.estimate_sensitivity` varies assumed demand by -20%, baseline, and +20% with the ACO seed fixed, reporting the same four study evaluation items. This demonstrates sensitivity to assumptions; it does not validate them against ground truth.

The app reports only distance traveled, path cost, computational time, and convergence rule as evaluations. ACO convergence does not establish global optimality. No TomTom or HERE account is required, and no external traffic observations are claimed.
