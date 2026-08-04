import pytest

from src.core.config import ConfigError, load_settings

_ENV_VARS_TO_CLEAR = (
    "OPENAI_MODEL",
    "ANTHROPIC_MODEL",
    "PERPLEXITY_MODEL",
    "REVISION_LOOP_CAP",
    "RESEARCH_TOOL_ITERATIONS_CAP",
    "MAX_RETRIES",
    "BACKOFF_SECONDS",
    "SESSION_STORE_URL",
    "SESSION_STORE_BACKEND",
    "SESSION_STORE_PATH",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "PERPLEXITY_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    # Never let the real .env file leak into these tests — full control via
    # monkeypatch only.
    monkeypatch.setattr("src.core.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-serp-key")
    for var in _ENV_VARS_TO_CLEAR:
        monkeypatch.delenv(var, raising=False)
    yield


def test_env_overrides_yaml_model(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    settings = load_settings("development")
    assert settings.openai_model == "gpt-4o-mini"


def test_yaml_default_used_when_no_env_override():
    settings = load_settings("development")
    assert settings.openai_model == "gpt-4o"  # from config/development.yaml


def test_missing_required_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        load_settings("development")


def test_revision_loop_cap_env_override(monkeypatch):
    monkeypatch.setenv("REVISION_LOOP_CAP", "5")
    settings = load_settings("development")
    assert settings.revision_loop_cap == 5


def test_revision_loop_cap_default_from_yaml():
    settings = load_settings("development")
    assert settings.revision_loop_cap == 1


def test_research_tool_iterations_cap_env_override(monkeypatch):
    monkeypatch.setenv("RESEARCH_TOOL_ITERATIONS_CAP", "7")
    settings = load_settings("development")
    assert settings.research_tool_iterations_cap == 7


def test_research_tool_iterations_cap_default_from_yaml():
    settings = load_settings("development")
    assert settings.research_tool_iterations_cap == 3


def test_production_session_store_defaults_to_postgres():
    settings = load_settings("production")
    assert settings.session_store_backend == "postgres"


def test_session_store_url_env_override(monkeypatch):
    monkeypatch.setenv("SESSION_STORE_URL", "postgresql://user:pass@example:5432/db")
    settings = load_settings("production")
    assert settings.session_store_url == "postgresql://user:pass@example:5432/db"


def test_session_store_backend_env_override(monkeypatch):
    monkeypatch.setenv("SESSION_STORE_BACKEND", "memory")
    settings = load_settings("production")
    assert settings.session_store_backend == "memory"


def test_session_store_path_env_override(monkeypatch, tmp_path):
    custom_path = str(tmp_path / "custom.db")
    monkeypatch.setenv("SESSION_STORE_PATH", custom_path)
    settings = load_settings("development")
    assert settings.session_store_path == custom_path


def test_session_store_path_defaults_relative_to_project_root():
    settings = load_settings("development")
    assert settings.session_store_path.endswith("contentalchemy.db")
