# Historical period with allowed estimates

Current period: **June 1, 2025 through May 31, 2026**, inclusive, Asia/Manila. Hourly resolution means 8,760 timestamps. The user now permits explicitly documented estimates. Earlier requirements for five years of exclusively non-estimated data are superseded.

The operational release combines historical provider-weather records with modeled traffic and static mapped road/flood proxies. It is not a fully observed historical dataset.

See [estimation methods and sources](estimation_methods.md) for all ten feature definitions, formulas, published references, project assumptions, and limitations. The active settings are `configs/historical_study.json` and `configs/study_assumptions.json`.

Weather timestamps are checked for completeness in each city's assigned grid. Provider measurement provenance and road-level representativeness are not independently verified. Missing weather records are rejected rather than silently filled. Traffic profiles, free-flow speeds, capacity, the time-of-day profile, and weather effects are explicitly estimated.

Road geometry and inventories from September 2026 are held constant as an estimated temporal surrogate. NOAH hazard maps are static exposure proxies, not hourly flood observations. User-selected closures and weather overrides are counterfactual scenarios.

The API verifies a release manifest before enabling routes. Readiness means the documented estimation pipeline is operational; it does not mean the model has been calibrated or validated against historical traffic measurements.
