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

    # Rule-based incident layer (src/routing/event_layer.py). It is NOT part of the thesis
    # method: there, event text reaches routing only through the learned DistilBERT ->
    # CNN+LSTM -> RADR STGNN risk scores, with W = distance * (1 + lambda * Risk). Off by
    # default so the app routes exactly as the thesis evaluation does; a value > 0 adds the
    # undocumented (1 + mu * Event) factor, for demos only. event_as_of pins the moment
    # incidents are evaluated at; empty means "use the risk snapshot's own window".
    event_mu: float = 0.0
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