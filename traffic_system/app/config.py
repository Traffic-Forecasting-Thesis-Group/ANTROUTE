from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://traffic:traffic@db:5432/traffic"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = "change-me-before-you-ship-anything" 
    access_token_expire_minutes: int = 60 * 24 * 7 
    google_places_api_key: str = ""

    # Scored risk_edges.csv from scripts/predict_congestion_risk.py. No live feed
    # exists, so a trip is routed on the recorded window matching its departure time
    # of day (see risk_routing.py). Not committed to git (same reasoning as embeddings.pt -- ~100MB) --
    # copy it here locally from wherever your scoring run wrote it, or override via
    # RISK_EDGES_PATH in .env.
    risk_edges_path: str = "data/processed/risk_scores/risk_edges.csv"

    # Event-Aware layer (src/routing/event_layer.py), part of the method: ANTROUTE's cost is
    # W = dist * (1 + lambda * Risk) * (1 + mu * Event), Event in [0, 1] from MMDA incident
    # reports (base impact 0.5 for one lane occupied, 1.0 for two or more, decaying linearly to
    # 0 over 60 minutes). event_mu is that mu; the default, 1.0, is the evaluated setting
    # (event_layer.DEFAULT_MU). 0 turns the penalty off -- the ablation, not the method.
    # event_as_of pins the moment incidents are evaluated at; empty means "use the risk
    # snapshot's own window".
    event_mu: float = 1.0
    event_as_of: str = ""

    # The baseline is Improved ACO (Cheng 2023) with dead-end recovery. When it still finds
    # no route the request says so (off, the default, as in the thesis evaluation). On, such
    # a trip gets the spatial shortest-distance route instead, labelled as such.
    baseline_fallback: bool = False

    # auto_labels.csv files (YOLO vehicle counts per frame) for the baseline's traffic-flow
    # term. Copy them here from the frames folders; without them the flow term is zero.
    vehicle_counts_dir: str = "data/processed/vehicle_counts"

    class Config:
        env_file = ".env"
        # This .env is shared with the data-ingestion side of the project (API keys
        # for Twitter/WeatherStack/etc., unrelated to this app), so unrecognized
        # vars must be ignored rather than rejected -- the default "forbid" makes
        # Settings() raise on startup whenever any of those keys is present.
        extra = "ignore"


settings = Settings()