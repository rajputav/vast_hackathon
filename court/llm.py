"""Court LLM — one chat() call over Weights & Biases serverless inference (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

BASE_URL: str = os.environ.get("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1").rstrip("/")
# Fastest model that reliably returns valid JSON in our tests (~0.5s); override per call or via env.
DEFAULT_MODEL: str = os.environ.get("WANDB_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
TIMEOUT_S: float = 60.0


def _load_env_file() -> None:
    """On the workshop VM the WANDB_* keys live in /etc/environment; pick them up if unset."""
    if os.environ.get("WANDB_API_KEY") or not os.path.exists("/etc/environment"):
        return
    with open("/etc/environment") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep and key.startswith("WANDB_"):
                os.environ.setdefault(key, value.strip().strip('"'))


def _project_header() -> str:
    """W&B wants "<team>/<project>". The k8s secret already stores it combined."""
    project = os.environ.get("WANDB_PROJECT", "")
    team = os.environ.get("WANDB_TEAM", "")
    if "/" in project or not team:
        return project
    return f"{team}/{project}"


def _complete(messages: list[dict[str, str]], model: str, json_mode: bool) -> str:
    _load_env_file()
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is not set")
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 1024}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Project": _project_header(),
            "Content-Type": "application/json",
            # Cloudflare in front of W&B rejects urllib's default UA with "error code: 1010".
            "User-Agent": "court/0.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"W&B inference HTTP {exc.code}: {exc.read()[:300]!r}") from exc
    content = data["choices"][0]["message"].get("content")
    if not content:
        # Reasoning-style models can spend the whole budget thinking and return no content.
        raise RuntimeError(f"{model} returned no content")
    return content.strip()


def _parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def chat(
    system: str,
    user: str,
    json_schema: dict | None = None,
    model: str | None = None,
) -> str | dict:
    """Send one system + user turn and return the reply.

    Without ``json_schema`` returns the reply text. With it, the model is told to return
    only a JSON object matching the schema; the parsed dict is returned. A reply that
    doesn't parse is retried once, then ``ValueError`` is raised.
    """
    model = model or DEFAULT_MODEL
    if json_schema is None:
        return _complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model,
            json_mode=False,
        )

    system = (
        f"{system}\n\nRespond with ONLY a JSON object (no prose, no code fences) that conforms "
        f"to this JSON Schema:\n{json.dumps(json_schema)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    reply = _complete(messages, model, json_mode=True)
    try:
        return _parse_json(reply)
    except ValueError as first_error:  # json.JSONDecodeError is a ValueError
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": f"That was not valid JSON ({first_error}). Reply again with only the JSON object."},
        ]
        reply = _complete(messages, model, json_mode=True)
        try:
            return _parse_json(reply)
        except ValueError as exc:
            raise ValueError(f"{model} did not return valid JSON after a retry: {reply[:200]!r}") from exc
