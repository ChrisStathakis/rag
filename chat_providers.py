"""chat_providers.py -- non-Ollama chat backends for Ask.

One function, stdlib only (urllib): `complete()` sends the same
system+user prompt Ask builds for Ollama to a hosted provider and returns
the answer text.

Providers
---------
- ``anthropic`` -- Anthropic Messages API (Claude).
- ``openai``    -- OpenAI Chat Completions.
- ``opencode``  -- OpenCode gateway (OpenAI Chat Completions shape).
- ``thegrid``   -- The Grid spot market (OpenAI Chat Completions shape).
- ``custom``    -- any OpenAI-compatible endpoint (base URL from Settings).

The last four all speak OpenAI Chat Completions, so they share one code
path; only the base URL differs. Model names are NOT hardcoded -- the
caller passes whatever model string the user typed. API keys come from
Settings (never logged).

Two gateway quirks are handled here:

- **307 redirects** (The Grid returns one on the way to its routing
  layer). urllib refuses to re-POST on 307 and would raise, so `_post`
  follows 3xx itself, preserving the method and body.
- Clear messages for 401 (bad key), 402 (out of credits) and 404 (wrong
  model: The Grid wants an *instrument* string, OpenCode needs an
  OpenAI-format model).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# provider id -> OpenAI-compatible base URL (path is appended per call)
PROVIDERS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "opencode": "https://opencode.ai/inference/openai/v1",
    "thegrid": "https://api.thegrid.ai/v1",
}

DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 1024
MAX_REDIRECTS = 5
_REDIRECT_CODES = (301, 302, 303, 307, 308)


class ProviderError(RuntimeError):
    """Expected failure (bad key, network, empty answer). Shown in the UI."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Stop urllib from silently turning a POST into a GET on 3xx.

    We follow redirects ourselves in `_post` so the method and JSON body
    survive (The Grid 307s on every request).
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None                                  # -> http_error_302 raises


_OPENER = urllib.request.build_opener(_NoRedirect)


def complete(provider: str, *, model: str, api_key: str, system: str,
             user: str, timeout: int = DEFAULT_TIMEOUT,
             max_tokens: int = DEFAULT_MAX_TOKENS,
             history: list | None = None, base_url: str = "") -> str:
    """Ask a hosted model. Raises ProviderError on any failure.

    `history` is [{"role": "user"|"assistant", "content": str}, ...] --
    previous exchanges, inserted before the current user message.
    `base_url` is only used by the ``custom`` provider.
    """
    model = (model or "").strip()
    if not model:
        raise ProviderError("no model name given")
    key = (api_key or "").strip()
    if not key:
        raise ProviderError("no API key -- set it in Settings")
    hist = [m for m in (history or [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and (m.get("content") or "").strip()]

    if provider == "anthropic":
        return _anthropic(model, key, system, user,
                          timeout, max_tokens, hist)
    if provider in PROVIDERS:
        return _openai_compatible(PROVIDERS[provider], model, key, system,
                                  user, timeout, max_tokens, hist, provider)
    if provider == "custom":
        return _openai_compatible(base_url, model, key, system,
                                  user, timeout, max_tokens, hist, "custom")
    raise ProviderError(f"unknown provider: {provider}")


def _http_message(code: int, detail: str, url: str) -> str:
    """Turn an HTTP status into an actionable one-liner."""
    detail = (detail or "").strip()
    if code in (401, 403):
        return f"HTTP {code}: bad or missing API key ({detail})"
    if code == 402:
        return f"HTTP 402: out of credits/balance ({detail})"
    if code == 404:
        return (f"HTTP 404: model or endpoint not found ({detail}). Check the "
                "model name -- The Grid wants an instrument string like "
                "'text-prime'; OpenCode needs a Chat-Completions model.")
    if code == 429:
        return f"HTTP 429: rate limited, try again ({detail})"
    return f"HTTP {code}: {detail or 'request failed'}"


def _post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    """POST JSON, following 3xx manually so the POST body is preserved."""
    data = json.dumps(payload).encode("utf-8")
    for _hop in range(MAX_REDIRECTS + 1):
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST")
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                if e.code in _REDIRECT_CODES:
                    loc = e.headers.get("Location") if e.headers else None
                    if loc:
                        url = urllib.parse.urljoin(url, loc)
                        continue
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:                        # noqa: BLE001
                    detail = ""
                raise ProviderError(_http_message(e.code, detail, url))
            finally:
                try:
                    e.close()
                except Exception:                        # noqa: BLE001
                    pass
        except Exception as e:                           # noqa: BLE001
            raise ProviderError(f"request failed: {e}")
    raise ProviderError("too many redirects")


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


def _openai_compatible(base_url: str, model: str, key: str, system: str,
                       user: str, timeout: int, max_tokens: int,
                       history: list, provider: str) -> str:
    """Shared backend for OpenAI and every OpenAI-compatible gateway."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ProviderError("no base URL set -- fill it in Settings")
    resp = _post(f"{base}/chat/completions",
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
        raise ProviderError(f"empty answer from {provider}")
    return text.strip()
