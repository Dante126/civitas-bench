"""Minimal OpenAI-compatible chat client (stdlib only).

Works with Arcee (https://api.arcee.ai/api/v1), OpenRouter, vLLM, Ollama,
OpenAI, etc. — anything exposing POST {base_url}/chat/completions.
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    base_url: str = "https://api.arcee.ai/api/v1"
    api_key_env: str = "ARCEE_API_KEY"
    timeout: float = 300.0        # give up when no data arrives for this long
    max_retries: int = 5
    extra_headers: dict | None = None
    stream: bool = True           # stream the answer: long reasoning turns never go quiet on the wire
    deadline: float = 1800.0      # wall-clock limit per attempt, however the server behaves


class LLMError(RuntimeError):
    pass


def _read_events(resp, idle: float) -> dict:
    """Reassemble a server-sent-events answer into the shape of a non-streamed one.

    Keep-alive comments reset the socket timeout, so a stream whose generation has stalled is caught
    here: no data line for `idle` seconds, or a stream that ends without a finish reason, is an error
    (and the attempt is retried) rather than a truncated answer.
    """
    content, reasoning, usage, finish, done = [], [], {}, None, False
    calls: dict[int, dict] = {}
    last = time.time()
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            if time.time() - last > idle:
                raise TimeoutError(f"stream stalled: no data for {idle:.0f} s")
            continue
        last = time.time()
        data = line[5:].strip()
        if data == "[DONE]":
            done = True
            break
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            for tc in delta.get("tool_calls") or []:       # fragments keyed by index, as OpenAI streams them
                slot = calls.setdefault(tc.get("index", 0),
                                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                slot["id"] = tc.get("id") or slot["id"]
                slot["type"] = tc.get("type") or slot["type"]
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            finish = ch.get("finish_reason") or finish
    if finish is None and not done:
        raise ConnectionError("stream ended before the answer was finished")
    message = {"content": "".join(content), "reasoning_content": "".join(reasoning)}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


def _post(req: urllib.request.Request, idle: float, deadline: float) -> dict:
    """POST and return the parsed answer (streamed or plain JSON).

    `idle` is urllib's socket timeout (fires only when no data arrives); `deadline` caps the whole
    attempt, because a non-streamed request can be held open for a very long time without an answer
    (seen with Arcee on generations longer than ~100 s).
    """
    box: dict = {}

    def work():
        try:
            with urllib.request.urlopen(req, timeout=idle) as resp:
                if "event-stream" in (resp.headers.get("Content-Type") or ""):
                    box["payload"] = _read_events(resp, idle)
                else:                        # providers / test servers that ignore "stream"
                    box["payload"] = json.loads(resp.read().decode())
        except BaseException as e:           # re-raised in the caller's thread
            box["err"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise TimeoutError(f"no complete response after {deadline:.0f} s")
    if "err" in box:
        raise box["err"]
    return box["payload"]


class ChatClient:
    def __init__(self, provider: ProviderConfig):
        self.p = provider
        self.api_key = os.environ.get(provider.api_key_env, "") if provider.api_key_env else ""

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.4,
             max_tokens: int = 1500, extra: dict | None = None) -> dict:
        """Returns {"content", "reasoning", "tool_calls", "usage", "latency_s", "finish_reason"}.

        Thinking models (e.g. trinity-large-thinking) may return their chain of
        thought in `reasoning_content` or inline as <think>...</think>.
        """
        if self.p.api_key_env and not self.api_key:
            raise LLMError(f"environment variable {self.p.api_key_env} is not set")
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, **(extra or {})}
        if self.p.stream:
            body.setdefault("stream", True)
            body.setdefault("stream_options", {"include_usage": True})
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
                payload = _post(req, self.p.timeout, self.p.deadline)
                choice = payload["choices"][0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                return {"content": content, "reasoning": reasoning, "tool_calls": msg.get("tool_calls") or [],
                        "usage": payload.get("usage") or {},
                        "latency_s": round(time.time() - t0, 3),
                        "finish_reason": choice.get("finish_reason")}
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise LLMError(last_err) from e
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, KeyError,
                    IndexError) as e:
                last_err = f"{type(e).__name__}: {e}"
            if attempt < self.p.max_retries:          # visible in the run log: slow runs are otherwise opaque
                print(f"[llm] {model}: attempt {attempt + 1} failed after {time.time() - t0:.0f} s ({last_err[:160]}); "
                      "retrying", file=sys.stderr, flush=True)
            time.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise LLMError(f"giving up after {self.p.max_retries + 1} attempts: {last_err}")
