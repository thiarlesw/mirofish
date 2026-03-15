"""
LLM client wrapper.
Uses OpenAI-compatible API format.

Fallback chain: primary models (DashScope/Qwen) → Gemini as last resort.
Configure via env vars:
  LLM_API_KEY        — DashScope API key
  LLM_BASE_URL       — DashScope base URL
  LLM_MODEL_NAME     — primary model (e.g. qwen-plus)
  LLM_FALLBACK_MODELS — comma-separated extra models on same provider (e.g. qwen-turbo)
  LLM_GEMINI_FALLBACK — Gemini model as last resort (uses GEMINI_API_KEY)
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI, RateLimitError, APIError

from ..config import Config

logger = logging.getLogger('mirofish.llm_client')

_GEMINI_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/'


class LLMClient:
    """LLM client with multi-provider fallback support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.primary_model = model or Config.LLM_MODEL_NAME

        # Primary client (DashScope or whatever LLM_BASE_URL points to)
        self._primary_client = OpenAI(
            api_key=api_key or Config.LLM_API_KEY,
            base_url=base_url or Config.LLM_BASE_URL,
        )

        # Gemini client as last-resort fallback
        self._gemini_client = None
        self._gemini_model = Config.LLM_GEMINI_FALLBACK
        if self._gemini_model and Config.GEMINI_API_KEY:
            self._gemini_client = OpenAI(
                api_key=Config.GEMINI_API_KEY,
                base_url=_GEMINI_BASE_URL,
            )

        self.fallback_models = Config.LLM_FALLBACK_MODELS

        if not (api_key or Config.LLM_API_KEY):
            raise ValueError("LLM_API_KEY is not configured")

    @property
    def model(self) -> str:
        return self.primary_model

    def _chain(self) -> List[Tuple[OpenAI, str]]:
        """Returns ordered list of (client, model) to try."""
        entries = [(self._primary_client, self.primary_model)]
        for m in self.fallback_models:
            if m != self.primary_model:
                entries.append((self._primary_client, m))
        if self._gemini_client and self._gemini_model:
            entries.append((self._gemini_client, self._gemini_model))
        return entries

    def _call(self, client: OpenAI, model: str, **kwargs) -> str:
        kwargs["model"] = model
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Some models (e.g. MiniMax M2.5) include <think>...</think> in content; strip it
        content = re.sub(r'<think>[\s\S]*?</think>', '', content).strip()
        return content

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """Send chat request with automatic provider/model fallback."""
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        chain = self._chain()
        last_error = None
        for client, model in chain:
            try:
                return self._call(client, model, **kwargs)
            except (RateLimitError, APIError) as e:
                last_error = e
                if (client, model) != chain[-1]:
                    logger.warning(f"Model {model} failed ({type(e).__name__}), trying next fallback")
        raise last_error
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Send chat request and return parsed JSON. Args: messages, temperature, max_tokens."""
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # Strip markdown code block markers
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM returned invalid JSON: {cleaned_response}")

