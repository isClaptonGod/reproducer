from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

from .types import Cost


def http_json(url: str, body: dict, headers: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json", **headers}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Response bodies can echo credentials. Never include them in logs.
        code = exc.code
        exc.close()
        raise RuntimeError(f"API returned HTTP {code}; check endpoint/access/model ID") from None
    except urllib.error.URLError:
        raise RuntimeError("API connection failed; check endpoint/network") from None


@dataclass
class Reply:
    text: str
    cost: Cost


class Provider:
    def __init__(self, config: dict):
        self.config = config
        self.kind = config["provider"]
        self.model = config["model"]
        self.observed_models: set[str] = set()
        if any(k in config for k in ("api_key", "token", "authorization")):
            raise ValueError("Use api_key_env, never inline credentials in provider configuration")
        if any(k in config.get("extra_body", {}) for k in ("model", "messages", "system", "seed")):
            raise ValueError("extra_body cannot override model, messages, system, or seed")
        self.base = config.get("base_url", "").rstrip("/")
        if config.get("base_url_env"):
            self.base = os.environ.get(config["base_url_env"], self.base).rstrip("/")
        if not self.base:
            raise ValueError(f"Missing API base URL for {self.model}")
        if urlparse(self.base).username or urlparse(self.base).password:
            raise ValueError("API URLs must not contain credentials")
        self.key = os.environ.get(config.get("api_key_env", ""), "")
        if config.get("require_key", True) and not self.key:
            raise ValueError(f"Set {config['api_key_env']} securely before running {self.model}")
        if self.kind not in {"openai", "anthropic"}:
            raise ValueError(f"Unsupported provider: {self.kind}")

    def _headers(self) -> dict:
        if self.kind == "anthropic":
            return {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def _body(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "messages": messages}
        if self.kind == "anthropic":
            body["system"] = messages[0]["content"]
            body["messages"] = messages[1:]
        return body

    def count_tokens(self, messages: list[dict]) -> int:
        if self.config.get("token_counter", "bytes") == "bytes":
            # Conservative byte upper bound, NOT comparable to the paper's tokenizer budget.
            return len(json.dumps(messages, ensure_ascii=False).encode())
        body = self._body(messages)
        if self.kind == "anthropic":
            return int(
                http_json(self.base + "/messages/count_tokens", body, self._headers())[
                    "input_tokens"
                ]
            )
        # vLLM's chat tokenization endpoint. Other servers must supply a compatible endpoint.
        url = self.config.get("tokenize_url") or self.base.removesuffix("/v1") + "/tokenize"
        result = http_json(url, body, self._headers())
        return int(result["count"])

    def complete(self, messages: list[dict], seed: int) -> Reply:
        body = self._body(messages)
        body.update(
            max_tokens=self.config.get("max_output_tokens", 4096),
            temperature=self.config.get("temperature", 0.7),
        )
        body.update(self.config.get("extra_body", {}))
        if self.kind == "openai" and self.config.get("send_seed", True):
            body["seed"] = seed
        endpoint = "/messages" if self.kind == "anthropic" else "/chat/completions"
        result = http_json(self.base + endpoint, body, self._headers())
        if result.get("model"):
            self.observed_models.add(result["model"])
            if self.config.get("strict_model_id", True) and result["model"] != self.model:
                raise RuntimeError(
                    "Provider reported a different model ID; configure an explicit "
                    "served alias or inspect model routing"
                )
        usage = result.get("usage")
        if usage is None:
            raise RuntimeError("Provider omitted usage; matched-cost experiments require usage")
        if self.kind == "anthropic":
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            cost = Cost(
                calls=1,
                input_tokens=usage["input_tokens"] + cache_read + cache_write,
                output_tokens=usage["output_tokens"],
                cached_tokens=cache_read,
                cache_write_tokens=cache_write,
            )
            content = "\n".join(x["text"] for x in result["content"] if x["type"] == "text")
        else:
            cost = Cost(
                calls=1,
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                cached_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            )
            content = result["choices"][0]["message"].get("content") or ""
        prices = self.config.get("usd_per_million_tokens", {})
        uncached = cost.input_tokens - cost.cached_tokens - cost.cache_write_tokens
        cost.usd = (
            uncached * prices.get("input", 0)
            + cost.cached_tokens * prices.get("cache_read", prices.get("input", 0))
            + cost.cache_write_tokens * prices.get("cache_write", prices.get("input", 0))
            + cost.output_tokens * prices.get("output", 0)
        ) / 1e6
        return Reply(content, cost)


class DemoProvider:
    """Deterministic plumbing fixture; never represents either requested real model."""

    config = {"max_output_tokens": 256, "token_counter": "bytes"}
    model = "scripted-fixture"

    def count_tokens(self, messages):
        return len(json.dumps(messages).encode())

    def complete(self, messages, seed):
        # Both branches inspect the artifact and improve it. No condition-specific oracle.
        last = messages[-1]["content"]
        try:
            observation = json.loads(last)
        except (ValueError, TypeError):
            observation = {}
        if observation.get("path") == "solution.json" and "content" in observation:
            current = json.loads(observation["content"])["value"]
            action = {
                "tool": "write",
                "path": "solution.json",
                "content": json.dumps({"value": min(10, current + 1)}),
            }
        else:
            action = {"tool": "read", "path": "solution.json"}
        text = json.dumps(action)
        return Reply(
            text,
            Cost(
                calls=1,
                input_tokens=len(json.dumps(messages)) // 4,
                output_tokens=max(1, len(text) // 4),
            ),
        )
