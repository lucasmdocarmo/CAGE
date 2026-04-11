"""Gemini adapter implementing InferenceEngine.

Uses Google Generative Language API (text-only) via REST.
- Requires env var GOOGLE_API_KEY
- Supports temperature, top_p, max_tokens
TTFT is approximated from wall-clock (no streaming available here).
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import requests

from .engine import InferenceEngine, InferenceRequest, InferenceResponse


class GeminiAdapter(InferenceEngine):
    """InferenceEngine adapter for the Gemini REST API (text-only)."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_base: str = "https://generativelanguage.googleapis.com/v1beta",
        timeout: int = 120,
        **kwargs,
    ):
        """Initialize the Gemini client.

        Raises:
            RuntimeError: If GOOGLE_API_KEY is not set.
        """
        super().__init__(model_name, **kwargs)
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        """Generate a response using Gemini.

        Note:
            The REST endpoint used here is non-streaming; `stream` is accepted
            for interface compatibility but ignored.
        """
        start_time = time.time()

        url = f"{self.api_base}/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "temperature": request.temperature,
                "topP": request.top_p,
                "maxOutputTokens": request.max_tokens,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            total_time_ms = (time.time() - start_time) * 1000

            candidates = data.get("candidates", [])
            text = ""
            finish_reason = "stop"
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join([p.get("text", "") for p in parts])
                finish_reason = candidates[0].get("finishReason", "stop")

            return InferenceResponse(
                request_id=request.request_id,
                generated_text=text,
                ttft_ms=0.0,  # not available via non-streaming Gemini API
                total_time_ms=total_time_ms,
                num_tokens=len(text.split()),
                model_name=self.model_name,
                finish_reason=finish_reason,
                error=None,
            )
        except requests.exceptions.RequestException as e:
            total_time_ms = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                ttft_ms=0.0,
                total_time_ms=total_time_ms,
                num_tokens=0,
                model_name=self.model_name,
                finish_reason="error",
                error=str(e),
            )

    def batch_generate(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        return [self.generate(r) for r in requests]

    def is_ready(self) -> bool:
        return True  # assumes API key is set; availability checked at call time

    def shutdown(self) -> None:
        pass
