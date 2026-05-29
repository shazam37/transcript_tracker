"""
LLM provider abstraction — change LLM_PROVIDER in .env, nothing else.

Supported providers:
  groq      → ChatGroq       (default: llama-3.1-70b-versatile or any Groq model)
  anthropic → ChatAnthropic  (default: claude-sonnet-4-6)
  openai    → ChatOpenAI     (default: gpt-4o)
  gemini    → ChatGoogleGenerativeAI (default: gemini-1.5-pro)

.env keys used:
  LLM_PROVIDER=groq                         # which provider to use
  LLM_MODEL=llama3-groq-70b-8192-tool-use-preview  # model name (optional override)
  GROQ_API_KEY=gsk_...
  ANTHROPIC_API_KEY=sk-ant-...
  OPENAI_API_KEY=sk-...
  GOOGLE_API_KEY=...

Every provider returns a LangChain BaseChatModel so agents are
completely unaware of which backend they're talking to.
"""
from __future__ import annotations
import os
from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel

# Provider → (env key for api_key, langchain class import path)
_PROVIDER_DEFAULTS: dict[str, dict] = {
    "groq":      {"default_model": "llama3-groq-70b-8192-tool-use-preview"},
    "anthropic": {"default_model": "claude-sonnet-4-6"},
    "openai":    {"default_model": "gpt-4o"},
    "gemini":    {"default_model": "gemini-1.5-pro"},
}


def get_llm(
    provider: str | None = None,
    model:    str | None = None,
    temperature: float = 0,
    max_tokens:  int   = 4096,
) -> BaseChatModel:
    """
    Build and return a LangChain chat model for the configured provider.

    Priority order for provider:
      1. Explicit `provider` argument (used in tests)
      2. LLM_PROVIDER env var
      3. Falls back to 'groq'

    Priority order for model name:
      1. Explicit `model` argument
      2. LLM_MODEL env var
      3. Provider default (see _PROVIDER_DEFAULTS)
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()

    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. "
            f"Choose from: {list(_PROVIDER_DEFAULTS)}"
        )

    model = model or os.getenv("LLM_MODEL") or _PROVIDER_DEFAULTS[provider]["default_model"]

    # ── Groq ──────────────────────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key_val = os.getenv("GROQ_API_KEY") or "placeholder"
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=api_key_val,
        )

    # ── Anthropic ─────────────────────────────────────────────────────────
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    # ── OpenAI ────────────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ── Gemini ────────────────────────────────────────────────────────────
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )


def current_provider_info() -> dict:
    """Returns the active provider + model for logging/display."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    model    = os.getenv("LLM_MODEL") or _PROVIDER_DEFAULTS.get(provider, {}).get("default_model", "unknown")
    return {"provider": provider, "model": model}