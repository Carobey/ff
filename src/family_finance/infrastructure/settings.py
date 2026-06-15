"""
Centralized settings. pydantic-settings 2.x — type-safe env vars.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Все настройки приложения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # === Сервис ===
    service_name: str = "family-finance"
    environment: str = "dev"
    log_level: str = "INFO"

    # === Telegram ===
    telegram_bot_token: SecretStr = SecretStr("put-your-token-here")
    telegram_allowed_user_ids: str = Field(default="")
    # Лимит размера загружаемого файла (выписка/фото чека) — disk-fill guard на границе.
    max_upload_mb: int = 20
    # Dev-воркэраунд: форсировать IPv4 + публичный DNS для сессии Telegram. Нужно
    # только в сетях, где локальный резолвер отдаёт нероутируемый IPv6 для
    # api.telegram.org. По умолчанию выключено (в проде сеть нормальная).
    telegram_force_ipv4_dns: bool = False

    @property
    def allowed_user_ids(self) -> set[int]:
        if not self.telegram_allowed_user_ids:
            return set()
        try:
            return {int(x.strip()) for x in self.telegram_allowed_user_ids.split(",") if x.strip()}
        except ValueError as exc:
            msg = (
                "TELEGRAM_ALLOWED_USER_IDS должен быть списком целых через запятую, "
                f"получено: {self.telegram_allowed_user_ids!r}"
            )
            raise ValueError(msg) from exc

    # === LLM: OpenRouter ===
    openrouter_api_key: SecretStr = SecretStr("sk-or-v1-replace-me")
    # Опциональные заголовки для OpenRouter leaderboard (https://openrouter.ai/docs)
    openrouter_http_referer: str | None = None
    openrouter_x_title: str | None = None

    # Модели в формате OpenRouter: "<provider>/<model>"
    llm_supervisor_model: str = "openai/gpt-5.4"
    llm_worker_model: str = "google/gemini-2.5-flash"

    # Fallback цепочки (OpenRouter автоматически переключится при ошибках/недоступности)
    llm_supervisor_fallbacks: str = ""
    llm_worker_fallbacks: str = ""

    @property
    def supervisor_fallback_list(self) -> list[str]:
        return [m.strip() for m in self.llm_supervisor_fallbacks.split(",") if m.strip()]

    @property
    def worker_fallback_list(self) -> list[str]:
        return [m.strip() for m in self.llm_worker_fallbacks.split(",") if m.strip()]

    # === ProverkaCheka (ФНС receipt API) ===
    proverkacheka_api_token: SecretStr | None = None

    # === Categorizer ===
    # False (default): LLM только для UNCLASSIFIED/needs_review.
    # True: прогнать ВСЕ транзакции через LLM (дороже, точнее для демо).
    llm_categorize_all: bool = False

    # Порог fuzzy-матча продавца к правилу-справочнику (pg_trgm word_similarity).
    # ≥ порога → категория берётся из БД без LLM. Ниже → уходит в LLM.
    merchant_match_threshold: float = 0.6

    # === Postgres ===
    # SecretStr: DSN содержит пароль — не должен светиться в repr(Settings)/логах.
    database_url: SecretStr = SecretStr(
        "postgresql://postgres:ff_dev_password@localhost:5432/family_finance"
    )
    database_url_async: SecretStr = SecretStr(
        "postgresql+asyncpg://postgres:ff_dev_password@localhost:5432/family_finance"
    )

    # === FalkorDB / Graphiti ===
    falkordb_host: str = "localhost"
    falkordb_port: int = 6380  # mapped from container 6379; see docker-compose.yml

    # === LangFuse ===
    langfuse_host: str = "http://localhost:3001"
    langfuse_public_key: SecretStr = SecretStr("pk-lf-dev")
    langfuse_secret_key: SecretStr = SecretStr("sk-lf-dev")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
