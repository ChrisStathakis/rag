"""chat_providers.py -- non-Ollama chat backends for Ask.

One function, stdlib only (urllib): `complete()` sends the same
system+user prompt Ask builds for Ollama to Anthropic (Claude) or
OpenAI (ChatGPT) and returns the answer text.

Model names are NOT hardcoded -- the caller passes whatever model string
the user typed. API keys come from Settings (never logged).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 1024


class ProviderError(RuntimeError):
    """Expected failure (bad key, network, empty answer). Shown in the UI."""


def complete(provider: str, *, model: str, api_key: str, system: str,
             user: str, timeout: int = DEFAULT_TIMEOUT,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             history: list | None = None) -> str:
    """Ask a hosted model. Raises ProviderError on any failure.

    `history` is [{"role": "user"|"assistant", "content": str}, ...] --
    previous exchanges, inserted before the current user message.
    """
    model = (model or "").strip()
    if not model:
        raise ProviderError("no model name given")
    if not (api_key or "").strip():
        raise ProviderError("no API key -- set it in Settings")
    hist = [m for m in (history or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()]
    if provider == "anthropic":
        return _anthropic(model, api_key.strip(), system, user,
                          timeout, max_tokens, hist)
    if provider == "openai":
        return _openai(model, api_key.strip(), system, user,
                       timeout, max_tokens, hist)
    raise ProviderError(f"unknown provider: {provider}")


def _post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:                                # noqa: BLE001
            detail = ""
        raise ProviderError(f"HTTP {e.code}: {detail or e.reason}")
    except Exception as e:                               # noqa: BLE001
        raise ProviderError(f"request failed: {e}")


def _anthropic(model: str, key: str, system: str, user: str,
               timeout: int, max_tokens: int,
               history: list) -> str:
    msgs = list(history)
    if msgs and msgs[-1].get("role") == "user":
        # merge trailing user turns so roles strictly alternate
        msgs[-1] = {"role": "user",
                    "content": msgs[-1].get("content", "") + "\n\n" + user}
    else:
        msgs.append({"role": "user", "content": user})
    resp = _post(ANTHROPIC_URL,
                 {"Content-Type": "application/json",
                  "x-api-key": key,
                  "anthropic-version": ANTHROPIC_VERSION},
                 {"model": model, "max_tokens": max_tokens,
                  "system": system,
                  "messages": msgs},
                 timeout)
    try:
        text = "".join(b.get("text", "") for b in resp.get("content", [])
                       if isinstance(b, dict) and b.get("type") == "text")
    except Exception:                                    # noqa: BLE001
        text = ""
    if not text.strip():
        raise ProviderError("empty answer from Anthropic")
    return text.strip()


def _openai(model: str, key: str, system: str, user: str,
            timeout: int, max_tokens: int,
            history: list) -> str:
    resp = _post(OPENAI_URL,
                 {"Content-Type": "application/json",
                  "Authorization": f"Bearer {key}"},
                 {"model": model, "max_tokens": max_tokens,
                  "messages": [{"role": "system", "content": system},
                               *history,
                               {"role": "user", "content": user}]},
                 timeout)
    try:
        text = resp["choices"][0]["message"]["content"] or ""
    except Exception:                                    # noqa: BLE001
        text = ""
    if not text.strip():
        raise ProviderError("empty answer from OpenAI")
    return text.strip()
