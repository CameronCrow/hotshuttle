"""Llama -- thin async client over the llama-server endpoints hotshuttle builds on.

Mechanism only: no policy, no model knowledge, no retries beyond transport. Everything
about *which* worker occupies *which* slot lives in pool.py / orchestrator.py.

Stdlib only. llama-server calls are I/O-bound and (with n_slots 1-2) barely concurrent,
so requests run on urllib in a worker thread via asyncio.to_thread rather than pulling in
aiohttp/httpx for two in-flight requests. ponytail: swap for a real async client only if
the slot count ever grows enough to make thread-per-request the bottleneck.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field


class LlamaError(RuntimeError):
    """The server answered, but not with success."""


@dataclass(frozen=True)
class Completion:
    """One /completion response, with the fields the layer above actually reasons about.

    prompt_n / cache_n are THE re-prefill telemetry and the reason this wrapper exists:
    the response also carries a `tokens_evaluated` field, which despite the name is the
    *full prompt length* whether or not any of it was cached. Asserting on that field can
    never fail -- measured in M1 at 2643 on a request that evaluated 28 tokens and reused
    2615. It is deliberately not surfaced here; use prompt_n.
    """
    content: str
    prompt_n: int          # prompt tokens actually evaluated this request
    cache_n: int           # prompt tokens reused from the slot's cache
    predicted_n: int       # tokens generated
    id_slot: int
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        """Full prompt length -- evaluated plus reused."""
        return self.prompt_n + self.cache_n

    @property
    def server_ms(self) -> float:
        """Time the server spent computing: prefill plus decode.

        This is the honest denominator for "was the GPU busy". Wall time inside dispatch
        is not -- it also contains queueing for a slot, and summing it across concurrent
        dispatches can exceed 100% of wall clock, which measures nothing.
        """
        return self.prompt_ms + self.predicted_ms

    @property
    def n_ctx_used(self) -> int:
        """Tokens resident in the slot after this turn."""
        return self.prompt_tokens + self.predicted_n


class Llama:
    def __init__(self, base_url: str, timeout: float = 900):
        # No default: which host and port the server lives on is profile data, not a
        # property of the client (docs/PLAN.md section 5).
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    # --- transport ---------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            raise LlamaError(f"{method} {path} -> {e.code}: {e.read()[:300]!r}") from e
        return json.loads(raw) if raw else {}

    async def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        return await asyncio.to_thread(self._request, method, path, body)

    # --- the four operations the pool needs --------------------------------

    async def complete(self, prompt: str, id_slot: int, n_predict: int,
                       cache_prompt: bool = True, **sampling) -> Completion:
        """Generate on a specific slot.

        cache_prompt=True makes the server diff `prompt` against the slot's cached tokens
        and evaluate only the unseen suffix -- which is why prompts must be append-only
        (invariant I3). Rewriting anything early silently invalidates from that byte on.
        """
        d = await self._call("POST", "/completion", {
            "prompt": prompt, "id_slot": id_slot, "n_predict": n_predict,
            "cache_prompt": cache_prompt, **sampling})
        t = d.get("timings", {})
        return Completion(content=d.get("content", ""),
                          prompt_n=t.get("prompt_n", 0), cache_n=t.get("cache_n", 0),
                          predicted_n=t.get("predicted_n", 0),
                          prompt_ms=t.get("prompt_ms", 0.0),
                          predicted_ms=t.get("predicted_ms", 0.0),
                          id_slot=d.get("id_slot", id_slot), raw=d)

    async def save(self, id_slot: int, filename: str) -> None:
        """Serialize a slot (attention KV + recurrent state) under --slot-save-path.

        Silently does nothing if the server was started without --slot-save-path, so
        bonsai.sh always passes it.
        """
        await self._call("POST", f"/slots/{id_slot}?action=save", {"filename": filename})

    async def restore(self, id_slot: int, filename: str) -> None:
        await self._call("POST", f"/slots/{id_slot}?action=restore", {"filename": filename})

    async def erase(self, id_slot: int) -> None:
        await self._call("POST", f"/slots/{id_slot}?action=erase", {})

    async def slots(self) -> list[dict]:
        return await self._call("GET", "/slots")

    async def tokenize(self, text: str) -> list[int]:
        return (await self._call("POST", "/tokenize", {"content": text}))["tokens"]

    async def healthy(self) -> bool:
        try:
            await self._call("GET", "/health")
            return True
        except Exception:
            return False
