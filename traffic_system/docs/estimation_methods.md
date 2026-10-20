# Estimation methods and sources

Period: **June 1, 2025 through May 31, 2026**, hourly, Asia/Manila. Scope: San Juan, Manila, Mandaluyong, Makati, and Pasig. Estimates are explicitly permitted. This is a routing model with historical provider-weather inputs, not a reconstructed archive of measured traffic on every road.

The executable settings are in [study_assumptions.json](../configs/study_assumptions.json). Every feature is disclosed in `data/processed/estimation_registry.json`, the network provenance, route responses, and the app's **Where each feature and estimate comes from** table.

## Ten-criterion basis (contract version 2)

The criteria, their formulas, units and encodings are defined once in [routing_contract.md](routing_contract.md) and
`src/routing/criteria.py`. The per-criterion classification, basis and source are in `src/routing/estimation.py`
(published as `data/processed/estimation_registry.json`). In short:

| Criterion | Formula | What is assumed or uncertain |
|---|---|---|
| Distance | L (km), WGS84 geodesic length | September 2026 geometry used as a static surrogate for the study year. |
| Congestion | L × g, g from the hourly table | An assumed, seeded congestion scenario (BPR on an assumed demand index); not observed or independently estimated demand. |
| Travel time | T (min) = free-flow time × BPR factor × assumed weather multiplier | Posted OSM speed is a free-flow proxy; fallback speeds are assumptions. |
| Intersections | I = 1 per traversed approach to a node with street count ≥ 3 | Topology held constant; geometry vertices never count. |
| Signals | S = 1 per traversed approach to a tagged signal | Untagged signals are assumed absent. |
| Road type | L × p, p from class median speeds | An assumed preference mapping, not a measured penalty. |
| Time profile | L × h, h fitted by road group, day category and hour on development dates | Fitted on the assumed scenario, so it is correlated with congestion. |
| Weather | L × a, a = assumed severity of the hourly category | Severities and speed multipliers are assumptions; kept separate from flood exposure. |
| Flood exposure | Σ km in each NOAH class × assumed class severity | Mapped susceptibility, not inundation or closure; class/3 severity is an assumed scenario with declared alternatives. |
| Capacity burden | L / K | A provisional study mapping; K uses an assumed 1,900 veh/h/lane base saturation flow. |

## Published functional forms

**BPR travel time.** The implementation uses `t = t0 * (1 + alpha*x^beta)`, where `x` is the assumed demand-to-capacity index, with reference coefficients `alpha=0.15`, `beta=4`. This supplies a functional form, not evidence for our assumed demand. [FHWA reference](https://www.fhwa.dot.gov/Environment/air_quality/conformity/research/sample_methodologies/emismeth03.cfm).

**Signalized capacity.** The reference relationship is `c = s * g/C`. The model uses `s = 1900` passenger-car-equivalent vehicles/hour/lane, assumes `g/C=0.5` at a mapped downstream signal, and otherwise uses 1. It does not model detailed lane movements, vehicle mix, or signal plans. Applying this simplified proxy to unsignalized links is a project extrapolation. [Capacity relationship](https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter3.htm) · [Saturation-flow reference](https://ops.fhwa.dot.gov/publications/fhwahop09008/chapter3.htm).

**Rain speed effects.** FHWA reports light-rain free-flow speed reductions of 2–13% and heavy-rain reductions of 6–17%. The model chooses the range midpoints, 7.5% and 11.5%, and converts speed reduction `r` to time multiplier `1/(1-r)`. These are transferred assumptions, not Philippine urban-road calibration. [FHWA road-weather reference](https://ops.fhwa.dot.gov/weather/roadimpact.htm).

Weather scenarios have categorical costs 0, 0.4, and 0.8. Historical precipitation/description chooses clear, rain, or heavy rain; the threshold of 7.5 mm in the provider precipitation field is a project classification rule. Neither categorical costs nor this threshold are claimed to come from the FHWA speed study. Weather does not invent flood closures.

## Project assumptions without empirical calibration

The configuration records all 24 hourly demand-to-capacity index values (`hourly_demand_capacity_ratio`). Each is multiplied by a road-type factor; weekend hours multiply demand by 0.75; repeatable variation (`traffic_variation`) is applied per city and day, per ~1.1 km area every three hours, and per road segment (static and three-hourly). `scripts/preprocess_traffic.py` evaluates this once for every segment and all 8,760 hours into `traffic_hourly.csv`. These numbers are scenario parameters chosen for the implementation, not reported Metro Manila traffic statistics. They must remain labeled as such.

Road-type fallback speeds, fallback lanes, categorical road costs, the assumed effective green ratio, and the weather flood-exposure share (0.5) are also provisional choices. Directional lane tags are preferred; two-way total lane tags are split between directions; otherwise the configured fallback is used. Unknown signals are assumed absent and reported as unrecorded.

The objective normalizes distance against 1,000 m, travel time against 600 s, and capacity against 4,000 equivalent vehicles/hour; it caps normalized values at one. These references are software settings, not measured input statistics.

## Criterion weights (hourly CRITIC)

The ten weights come from CRITIC (Diakoulaki, Mavrotas and Papayannakis, 1995, https://doi.org/10.1016/0305-0548(94)00059-H)
computed by `scripts/build_study_network.py` over development hours only (see [routing_contract.md](routing_contract.md#calibration-development-data-only)).

- In each sampled development hour, the alternatives are all road segments and the criteria are the ten additive criteria.
  A criterion that is constant in that hour gets weight 0 and is left out of every correlation and conflict sum.
- The fixed weight of each criterion is the mean of its hourly weights over one common valid-hour set, with zeros kept.
  Earlier versions divided each criterion by the number of hours in which it varied, which inflated rarely varying
  criteria (weather received 0.126 instead of about 0.064); that estimator is removed.
- Hours in which CRITIC is undefined (every criterion constant, or perfectly concordant varying criteria) or whose
  inputs are missing are excluded whole and logged in `data/processed/calibration.json`; equal weights are never invented.
- CRITIC weights are objective with respect to the modeled criterion values; they are not calibrated against observed route choices.

## Historical coverage and provenance

The source CSV has 34 grid cells, each with 8,760 hourly records. The builder verifies all expected timestamps and rejects duplicate or missing timestamps for the selected grids. It saves a source hash and the exact city-to-grid assignments in `data/processed/historical_weather.json`. Cities may share a grid; those are not independent city measurements. No live October 2026 weather is used for a study route.

Traffic varies by selected hour and weekday through the disclosed model. Historical weather varies by actual CSV timestamp. This does **not** create 8,760 independent observed traffic samples. Road geometry, classifications, signals, and mapped flood exposure are static surrogates.

Manually entered closures are counterfactual scenarios, not claims that those roads were actually closed on the selected date. Clear/rain/heavy-rain overrides are likewise labeled scenarios; the historical-weather option uses the dated provider records.

Mapped sources: [OSM attribution](https://www.openstreetmap.org/copyright), [OSMnx conversion convention](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.convert.graph_to_gdfs), [NOAH hazard methodology](https://noah.up.edu.ph/know-your-hazards). Weather comes from the retained `weatherstack_hourly.csv`; the provider's historical endpoint was separately verified to return hourly records, but API access does not establish measurement provenance.

## Verification and sensitivity

The release manifest hashes the network, weather bundle, assumptions, and period policy. Changed or missing artifacts block routing until the release is rebuilt. This checks artifact integrity, not scientific validity.

`python -m scripts.validate_system` checks real-network routing and closure exclusion. `python -m scripts.estimate_sensitivity` varies assumed demand by -20%, baseline, and +20% with the ACO seed fixed, reporting the same four study evaluation items. This demonstrates sensitivity to assumptions; it does not validate them against ground truth.

The app reports only distance traveled, path cost, computational time, and convergence rule as evaluations. ACO convergence does not establish global optimality. No TomTom or HERE account is required, and no external traffic observations are claimed.
