"""
llm_handler.py
--------------
Cloud API LLM support — OpenAI, Google Gemini, Anthropic Claude.
User provides their own API key in the UI.
"""

from typing import Any, Optional

from config import API_PROVIDERS
from utils import get_logger

logger = get_logger(__name__)


def get_llm(provider: str, model_name: str, api_key: str, temperature: float = 0.1) -> Any:
    """
    Create and return a LangChain LLM for the given provider.

    Args:
        provider:    "Google Gemini", "OpenAI", or "Anthropic Claude"
        model_name:  Model ID string
        api_key:     User's API key
        temperature: Lower = more deterministic answers

    Returns:
        LangChain-compatible chat model object
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key cannot be empty.")

    if provider not in API_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'.")

    logger.info("Creating LLM: provider=%s model=%s", provider, model_name)

    if provider == "Google Gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise RuntimeError("Run: pip install langchain-google-genai")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key.strip(),
            temperature=temperature,
        )

    elif provider == "OpenAI":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError("Run: pip install langchain-openai")
        return ChatOpenAI(
            model=model_name,
            api_key=api_key.strip(),
            temperature=temperature,
        )

    elif provider == "Anthropic Claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise RuntimeError("Run: pip install langchain-anthropic")
        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key.strip(),
            temperature=temperature,
        )


def generate_answer(llm: Any, prompt: str) -> str:
    """Send prompt to LLM and return text response."""
    try:
        logger.info("Sending prompt (%d chars) to LLM…", len(prompt))
        response = llm.invoke(prompt)
        # Chat models return AIMessage object — get .content
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info("Response received (%d chars).", len(answer))
        return answer.strip()
    except Exception as exc:
        raise RuntimeError(
            f"Error calling model '{llm.model}' ({type(exc).__name__}): {exc}"
        ) from exc
