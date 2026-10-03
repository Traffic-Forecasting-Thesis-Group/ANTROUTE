from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://traffic:traffic@db:5432/traffic"
    redis_url: str = "redis://redis:6379/0"
    secret_key: str = "change-me-before-you-ship-anything" 
    access_token_expire_minutes: int = 60 * 24 * 7 
    google_places_api_key: str = ""  

    class Config:
        env_file = ".env"


settings = Settings()