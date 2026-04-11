"""Ollama adapter implementing InferenceEngine.

Uses Ollama's local HTTP API:
POST /api/generate
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import requests

from .engine import InferenceEngine, InferenceRequest, InferenceResponse


class OllamaAdapter(InferenceEngine):
    """InferenceEngine adapter for the Ollama REST API."""

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:11434",
        timeout: int = 300,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.generate_url = f"{self.api_base}/api/generate"

    def _build_payload(self, request: InferenceRequest, *, stream: bool) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "num_predict": request.max_tokens,
        }
        if request.stop:
            options["stop"] = request.stop

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": request.prompt,
            "stream": stream,
            "options": options,
        }
        return payload

    def _stream_generate(self, request: InferenceRequest) -> InferenceResponse:
        start_time = time.time()
        first_token_time: Optional[float] = None
        full_text_parts: list[str] = []
        finish_reason = "length"

        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None

        router_replica = None
        try:
            with requests.post(
                self.generate_url,
                json=self._build_payload(request, stream=True),
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()
                router_replica = resp.headers.get("x-router-replica")

                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    if obj.get("response"):
                        full_text_parts.append(obj.get("response", ""))
                        if first_token_time is None:
                            first_token_time = time.time()

                    if obj.get("done"):
                        finish_reason = "stop"
                        prompt_tokens = obj.get("prompt_eval_count")
                        completion_tokens = obj.get("eval_count")
                        break
        except requests.exceptions.RequestException as e:
            total_time = (time.time() - start_time) * 1000
            return InferenceResponse(
                request_id=request.request_id,
                generated_text="",
                ttft_ms=0.0,
                total_time_ms=total_time,
                num_tokens=0,
                model_name=self.model_name,
                finish_reason="error",
                error=str(e),
            )

        total_time_ms = (time.time() - start_time) * 1000
        ttft_ms = ((first_token_time - start_time) * 1000) if first_token_time else total_time_ms
        generated_text = "".join(full_text_parts)
        num_tokens = (
            int(completion_tokens) if isinstance(completion_tokens, (int, float)) else len(generated_text.split())
        )

        return InferenceResponse(
            request_id=request.request_id,
            generated_text=generated_text,
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            num_tokens=num_tokens,
            model_name=self.model_name,
            finish_reason=finish_reason,
            router_replica=router_replica,
            prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
            cached_prompt_tokens=None,
        )

    def generate(self, request: InferenceRequest, *, stream: bool = False) -> InferenceResponse:
        if stream:
            return self._stream_generate(request)

        start_time = time.time()
        router_replica = None
        try:
            resp = requests.post(
                self.generate_url,
                json=self._build_payload(request, stream=False),
                timeout=self.timeout,
            )
            resp.raise_for_status()
            router_replica = resp.headers.get("x-router-replica")
            data = resp.json()
            total_time_ms = (time.time() - start_time) * 1000

            generated_text = data.get("response", "") if isinstance(data, dict) else ""
            finish_reason = "stop" if data.get("done") else "length"
            prompt_tokens = data.get("prompt_eval_count")
            completion_tokens = data.get("eval_count")

            num_tokens = (
                int(completion_tokens) if isinstance(completion_tokens, (int, float)) else len(generated_text.split())
            )
            ttft_ms = total_time_ms * 0.2

            return InferenceResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                num_tokens=num_tokens,
                model_name=self.model_name,
                finish_reason=finish_reason,
                router_replica=router_replica,
                prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None,
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
        return [self.generate(req) for req in requests]

    def is_ready(self) -> bool:
        try:
            resp = requests.get(f"{self.api_base}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def shutdown(self) -> None:
        pass
