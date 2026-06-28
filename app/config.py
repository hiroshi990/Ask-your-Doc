from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Cohere
    cohere_api_key: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Chunking
    chunk_max_tokens: int = 700
    chunk_min_tokens: int = 500
    chunk_overlap_tokens: int = 75

    # Retrieval
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    rerank_top_k: int = 5

    #Monitoring
    langsmith_api_key: str = ""

    # Paths
    upload_dir: Path = Path("./uploads")
    data_dir: Path = Path("./data")
    log_level: str = "INFO"

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings

def bust_cache():
    get_settings.cache_clear()
    return get_settings()