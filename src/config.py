from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

try:
    from model_provider import ProviderConfig
except ImportError:
    from .model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Configuration shared across the lab.

    Attributes:
        base_dir: Root directory of the repository.
        data_dir: Directory containing benchmark datasets.
        state_dir: Directory to persist state and user profiles (e.g. User.md).
        compact_threshold_tokens: Estimated token threshold to trigger compaction.
        compact_keep_messages: Number of recent messages to preserve during compaction.
        model: Configuration for the primary conversational model.
        judge_model: Configuration for the evaluator/judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _build_provider_config(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProviderConfig:
    """Builds a ProviderConfig instance, resolving provider-specific API keys and base URLs."""
    norm_provider = provider.strip().lower()
    if norm_provider in ("anthorpic",):
        norm_provider = "anthropic"
    elif norm_provider in ("google",):
        norm_provider = "gemini"

    resolved_api_key = api_key
    resolved_base_url = base_url

    if not resolved_api_key:
        if norm_provider == "openai":
            resolved_api_key = os.getenv("OPENAI_API_KEY")
        elif norm_provider == "gemini":
            resolved_api_key = os.getenv("GEMINI_API_KEY")
        elif norm_provider == "anthropic":
            resolved_api_key = os.getenv("ANTHROPIC_API_KEY")
        elif norm_provider == "openrouter":
            resolved_api_key = os.getenv("OPENROUTER_API_KEY")
        elif norm_provider == "custom":
            resolved_api_key = os.getenv("CUSTOM_API_KEY")

    if not resolved_base_url:
        if norm_provider == "ollama":
            resolved_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        elif norm_provider == "openrouter":
            resolved_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif norm_provider == "custom":
            resolved_base_url = os.getenv("CUSTOM_BASE_URL")

    return ProviderConfig(
        provider=norm_provider,
        model_name=model_name,
        temperature=temperature,
        api_key=resolved_api_key,
        base_url=resolved_base_url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Loads environment variables and returns a populated LabConfig instance."""
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load environment variables from .env if present
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    compact_threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "800"))
    compact_keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))

    # Primary model configuration
    main_provider = os.getenv("LLM_PROVIDER", "openai")
    main_model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    try:
        main_temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    except ValueError:
        main_temp = 0.0

    model_cfg = _build_provider_config(
        provider=main_provider,
        model_name=main_model_name,
        temperature=main_temp,
    )

    # Judge model configuration
    judge_provider = os.getenv("JUDGE_LLM_PROVIDER", main_provider)
    judge_model_name = os.getenv("JUDGE_LLM_MODEL", main_model_name)
    try:
        judge_temp = float(os.getenv("JUDGE_LLM_TEMPERATURE", "0.0"))
    except ValueError:
        judge_temp = 0.0

    judge_cfg = _build_provider_config(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=judge_temp,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold,
        compact_keep_messages=compact_keep,
        model=model_cfg,
        judge_model=judge_cfg,
    )
