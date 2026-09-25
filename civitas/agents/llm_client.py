"""Minimal OpenAI-compatible chat client (stdlib only).

Works with Arcee (https://api.arcee.ai/api/v1), OpenRouter, vLLM, Ollama,
OpenAI, etc. — anything exposing POST {base_url}/chat/completions.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    base_url: str = "https://api.arcee.ai/api/v1"
    api_key_env: str = "ARCEE_API_KEY"
    timeout: float = 300.0
    max_retries: int = 5
    extra_headers: dict | None = None


class LLMError(RuntimeError):
    pass


def _post(req: urllib.request.Request, timeout: float) -> bytes:
    """POST and read the whole body, giving up after `timeout` seconds of wall clock.

    urllib's own timeout only fires when the socket is silent; a provider that holds a request open
    while trickling bytes (seen with Arcee) would otherwise block a run for many minutes.
    """
    box: dict = {}

    def work():
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                box["body"] = resp.read()
        except BaseException as e:           # re-raised in the caller's thread
            box["err"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"no complete response after {timeout:.0f} s")
    if "err" in box:
        raise box["err"]
    return box["body"]


class ChatClient:
    def __init__(self, provider: ProviderConfig):
        self.p = provider
        self.api_key = os.environ.get(provider.api_key_env, "") if provider.api_key_env else ""

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.4,
             max_tokens: int = 1500, extra: dict | None = None) -> dict:
        """Returns {"content", "reasoning", "usage", "latency_s", "finish_reason"}.

        Thinking models (e.g. trinity-large-thinking) may return their chain of
        thought in `reasoning_content` or inline as <think>...</think>.
        """
        if self.p.api_key_env and not self.api_key:
            raise LLMError(f"environment variable {self.p.api_key_env} is not set")
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **(extra or {})}
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", **(self.p.extra_headers or {})}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = self.p.base_url.rstrip("/") + "/chat/completions"
        last_err = None
        for attempt in range(self.p.max_retries + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                payload = json.loads(_post(req, self.p.timeout).decode())
                choice = payload["choices"][0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                return {"content": content, "reasoning": reasoning, "usage": payload.get("usage") or {},
                        "latency_s": round(time.time() - t0, 3),
                        "finish_reason": choice.get("finish_reason")}
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise LLMError(last_err) from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError) as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise LLMError(f"giving up after {self.p.max_retries + 1} attempts: {last_err}")
