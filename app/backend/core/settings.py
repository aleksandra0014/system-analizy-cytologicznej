from pydantic import Field, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path

class Settings(BaseSettings):
    ENV: str = Field("dev", validation_alias="ENV")
    PORT: int = 8000
    CORS_ORIGINS: List[AnyHttpUrl] | List[str] = []

    MONGODB_URI: str = Field("mongodb://localhost:27017", alias="MONGO_URI")
    MONGODB_DB: str = Field("lbc_db2", alias="MONGO_DB")

    JWT_SECRET: str = "change-me-in-env"
    JWT_ALG: str = "HS256"
    ACCESS_TTL_SECONDS: int = 60 * 60 * 12
    COOKIE_NAME: str = "lbc_session"


    model_config = SettingsConfigDict(env_file=str(Path(__file__).resolve().parent / ".env"), case_sensitive=False)

    @property
    def cookie_secure(self) -> bool:
        return self.ENV.lower() == "prod"

    @property
    def cookie_samesite(self) -> str:
        return "none" if self.cookie_secure else "lax"

settings = Settings()
