from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_database: str = "default"
    milvus_timeout_seconds: int = 30

    # MCP server
    server_name: str = "mcp-milvus-server"
    server_host: str = "0.0.0.0"
    server_port: int = 8082
    mount_path: str = "/mcp"
    mcp_auth_token: str = "change-me"
    log_level: str = "INFO"

    # Safety
    read_only_mode: bool = True
    max_search_limit: int = 100
    max_concurrent_searches: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
