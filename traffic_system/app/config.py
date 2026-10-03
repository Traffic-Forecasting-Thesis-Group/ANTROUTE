from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://traffic:traffic@db:5432/traffic"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = "change-me-before-you-ship-anything" 
    access_token_expire_minutes: int = 60 * 24 * 7 
    google_places_api_key: str = ""

    # Scored risk_edges.csv from scripts/predict_congestion_risk.py. No live feed
    # exists, so this is always the newest available scored window standing in for
    # "right now". Not committed to git (same reasoning as embeddings.pt -- ~100MB) --
    # copy it here locally from wherever your scoring run wrote it, or override via
    # RISK_EDGES_PATH in .env.
    risk_edges_path: str = "data/processed/risk_scores/risk_edges.csv"

    class Config:
        env_file = ".env"
        # This .env is shared with the data-ingestion side of the project (API keys
        # for Twitter/WeatherStack/etc., unrelated to this app), so unrecognized
        # vars must be ignored rather than rejected -- the default "forbid" makes
        # Settings() raise on startup whenever any of those keys is present.
        extra = "ignore"


settings = Settings()