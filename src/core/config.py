"""Application configuration loading (env vars, config/*.yaml)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}")


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    environment: str

    openai_api_key: str
    anthropic_api_key: str
    google_api_key: str
    serpapi_api_key: str
    perplexity_api_key: str

    openai_model: str
    anthropic_model: str
    google_model: str
    perplexity_model: str

    llm_primary_provider: str
    llm_fallback_provider: str
    research_primary_provider: str
    research_fallback_provider: str
    image_primary_provider: str
    image_fallback_provider: str | None

    session_store_backend: str
    session_store_url: str | None
    session_store_path: str

    max_retries: int
    backoff_seconds: float
    revision_loop_cap: int
    research_tool_iterations_cap: int
    cache_ttl_minutes: int
    rate_limit_per_minute: int

    log_level: str


def _expand_env(value: object) -> object:
    """Recursively resolves ${VAR} placeholders against os.environ."""
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _load_yaml(name: str) -> dict:
    path = _CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _expand_env(data)


def _require(value: str | None, env_var: str) -> str:
    if not value:
        raise ConfigError(f"Required environment variable {env_var} is not set")
    return value


def load_settings(environment: str | None = None) -> Settings:
    load_dotenv()

    environment = environment or os.getenv("CONTENTALCHEMY_ENV", "development")
    env_yaml = _load_yaml(f"{environment}.yaml")
    services_yaml = _load_yaml("services.yaml")

    llm = env_yaml.get("llm", {})
    research = env_yaml.get("research", {})
    image = env_yaml.get("image", {})
    session_store = env_yaml.get("session_store", {})
    logging_cfg = env_yaml.get("logging", {})
    resilience = services_yaml.get("resilience", {})
    performance = services_yaml.get("performance", {})

    return Settings(
        environment=environment,
        openai_api_key=_require(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        serpapi_api_key=_require(os.getenv("SERPAPI_API_KEY"), "SERPAPI_API_KEY"),
        perplexity_api_key=os.getenv("PERPLEXITY_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL") or llm.get("primary_model", "gpt-4o"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", ""),
        google_model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
        perplexity_model=os.getenv("PERPLEXITY_MODEL", ""),
        llm_primary_provider=llm.get("primary_provider", "openai"),
        llm_fallback_provider=llm.get("fallback_provider", "anthropic"),
        research_primary_provider=research.get("primary_provider", "serpapi"),
        research_fallback_provider=research.get("fallback_provider", "perplexity"),
        image_primary_provider=image.get("primary_provider", "openai"),
        image_fallback_provider=image.get("fallback_provider"),
        session_store_backend=os.getenv("SESSION_STORE_BACKEND") or session_store.get("backend", "memory"),
        session_store_url=os.getenv("SESSION_STORE_URL") or session_store.get("url") or None,
        session_store_path=str(
            _PROJECT_ROOT
            / (os.getenv("SESSION_STORE_PATH") or session_store.get("path", "contentalchemy.db"))
        ),
        max_retries=int(os.getenv("MAX_RETRIES") or resilience.get("max_retries", 3)),
        backoff_seconds=float(os.getenv("BACKOFF_SECONDS") or resilience.get("backoff_seconds", 2)),
        revision_loop_cap=int(os.getenv("REVISION_LOOP_CAP") or resilience.get("revision_loop_cap", 1)),
        research_tool_iterations_cap=int(
            os.getenv("RESEARCH_TOOL_ITERATIONS_CAP") or resilience.get("research_tool_iterations_cap", 3)
        ),
        cache_ttl_minutes=int(performance.get("cache_ttl_minutes", 30)),
        rate_limit_per_minute=int(performance.get("rate_limit_per_minute", 60)),
        log_level=logging_cfg.get("level", "INFO"),
    )


@lru_cache
def get_settings() -> Settings:
    return load_settings()
