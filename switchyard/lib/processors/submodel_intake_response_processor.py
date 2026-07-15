# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit a routing strategy's own model calls to intake as separate records.

A routing strategy makes an extra model call on every turn to decide where to
send the request: the deterministic LLM classifier, the stage-router tier
classifier, and the plan/execute planner all do this. That call never reaches
the routed backend, so the Rust intake response processor — which logs one
record per routed turn — never sees it, and its token spend is absent from
NVDataflow even though it is tracked in the ``StatsAccumulator``
(``/v1/routing/stats``). Reported blended cost is then too low, and
savings-vs-baseline too high, by the per-turn routing overhead.

Each of those routers stashes its per-turn usage on
``ctx.metadata[CTX_SUBMODEL_CALLS]`` (a list, so several sub-calls in one chain
accumulate rather than overwrite). This processor emits one intake record per
entry. Only the model and token counts are sent — never the routing call's
input messages — so the record stays anonymous.

The record mirrors the primary intake record: emission is gated on the same
per-request capture opt-in (``RequestMetadata.intake.enabled``), the session id
comes from the same ``RequestMetadata``, and it reuses the payload
cost/version/timestamp helpers, so NVDataflow flattens a routing-call record the
same way it flattens a routed turn. It emits through its own ``IntakeClient`` —
the same client class and payload helpers the routed-turn path uses, but a
separate instance this processor builds lazily and closes in ``shutdown()``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from switchyard.lib.processors.intake_client import IntakeClient
from switchyard.lib.processors.intake_payload_builder import (
    CTX_SUBMODEL_CALLS,
    _cost_fields,
    _created_at_iso,
    _request_metadata,
    _switchyard_version,
)

if TYPE_CHECKING:
    from switchyard.lib.config.intake_sink_config import IntakeSinkConfig
    from switchyard.lib.proxy_context import ProxyContext
    from switchyard.lib.request_metadata import RequestMetadata
    from switchyard_rust.core import ChatResponse

log = logging.getLogger(__name__)
JsonObject = dict[str, object]


class SubModelIntakeResponseProcessor:
    """Emit a routing strategy's own model calls to intake per turn, fail-open.

    The ``IntakeClient`` is built lazily on first emission so construction
    stays cheap and a missing SDK can never break a turn. ``client`` can be
    injected for tests.
    """

    def __init__(
        self,
        config: IntakeSinkConfig,
        *,
        client: IntakeClient | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._owns_client = client is None
        self._disabled = False

    async def process(self, ctx: ProxyContext, response: ChatResponse) -> ChatResponse:
        """Emit one anonymous intake record per stashed routing-call, fail-open.

        Reads ``ctx.metadata[CTX_SUBMODEL_CALLS]`` and emits a record for each
        valid entry when the turn opted into capture. Returns ``response``
        unchanged; malformed entries and build/transport errors are skipped.
        """
        if self._disabled:
            return response
        calls = ctx.metadata.get(CTX_SUBMODEL_CALLS)
        if not isinstance(calls, list) or not calls:
            return response
        metadata = _request_metadata(ctx)
        # Emit only when this turn opted into capture, matching the primary
        # record's gate (the x-switchyard-intake-enabled header the launcher and
        # `serve --intake-enabled` send). Without this, routing-call records
        # would leak for callers who did not opt in.
        # Known gap: this mirrors the header opt-in only. The primary Rust sink
        # also treats an OpenAI `store: true` request body as opt-in, which this
        # Python path can't see (there is no request here), so a raw client using
        # only `store` under-emits its routing record — the safe direction (a
        # missing record, never a leaked one). The follow-up that moves emission
        # into the Rust sink inherits that sink's exact opt-in gate and closes it.
        if metadata.intake.enabled is not True:
            return response
        try:
            client = self._ensure_client()
        except Exception:
            log.exception("submodel intake: client unavailable; disabling")
            self._disabled = True
            return response
        for call in calls:
            if not isinstance(call, dict):
                continue
            # Fail-open twice over: skip an entry that lacks a usable model or
            # valid token counts (a synthetic zero-cost record helps no one), and
            # never let a build/transport error turn a successful LLM response
            # into a 500 (the response chain re-raises processor exceptions).
            try:
                payload = _build_submodel_payload(self._config, ctx, metadata, call)
                if payload is None:
                    continue
                client.enqueue_background(_payload_factory(payload))
            except Exception:
                log.exception("submodel intake: failed to build/emit a record; skipping")
        return response

    async def shutdown(self) -> None:
        """Close the ``IntakeClient`` if this processor built it (never an injected one)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    def _ensure_client(self) -> IntakeClient:
        """Return the intake client, building it lazily on first use."""
        if self._client is None:
            self._client = IntakeClient(self._config)
        return self._client


def _build_submodel_payload(
    config: IntakeSinkConfig,
    ctx: ProxyContext,
    metadata: RequestMetadata,
    call: dict[str, object],
) -> JsonObject | None:
    """Build an intake payload for one routing model call (no message content).

    Returns ``None`` for a malformed entry — no usable model, or a token count
    that isn't a non-negative int — so the caller skips it rather than emitting
    a synthetic zero-cost record.
    """
    model = call.get("model")
    if not isinstance(model, str) or not model:
        return None
    prompt = _nonneg_int(call.get("prompt_tokens"))
    completion = _nonneg_int(call.get("completion_tokens"))
    cached = _nonneg_int(call.get("cached_tokens", 0))
    if prompt is None or completion is None or cached is None:
        return None
    router_type = call.get("router_type")
    routed_to = call.get("routed_to")
    openai_response: JsonObject = {
        "model": model,
        "object": "chat.completion",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
    session_id = metadata.session_id
    session_id = session_id if isinstance(session_id, str) and session_id else None
    switchyard_meta: JsonObject = {
        "version": _switchyard_version(),
        "inbound_format": _inbound_format(ctx),
        "stream": False,
        "user_id": config.user_id,
        "created_at": _created_at_iso(None, None),
        "routing": {
            "router_type": router_type if isinstance(router_type, str) else "",
            "routed_to": routed_to if isinstance(routed_to, str) else "",
        },
    }
    if session_id is not None:
        switchyard_meta["session_id"] = session_id
    payload: JsonObject = {
        "request": {"model": model, "switchyard": switchyard_meta},
        "response": openai_response,
        "provider": "switchyard",
    }
    payload.update(_cost_fields(openai_response))
    if session_id is not None:
        payload["session_id"] = session_id
        # Match the primary record so a routing call joins the same eval run.
        payload["evaluation_context"] = {
            "evaluation_run_id": session_id,
            "test_case_id": metadata.intake.task or "chat",
        }
    payload["user_id"] = config.user_id
    return payload


def _inbound_format(ctx: ProxyContext) -> str | None:
    """The turn's inbound request format (e.g. ``openai_chat``), or ``None``."""
    fmt = getattr(ctx, "inbound_format", None)
    return str(fmt) if fmt is not None else None


def _payload_factory(payload: JsonObject) -> Callable[[], JsonObject]:
    """Bind ``payload`` into a zero-arg factory for ``enqueue_background``.

    A helper (not an inline ``lambda p=payload: p``) so each loop iteration
    captures its own payload rather than the last one, and so the closure's
    type is inferrable.
    """
    return lambda: payload


def _nonneg_int(value: object) -> int | None:
    """Return ``value`` if it is a non-negative int (not bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


__all__ = ["SubModelIntakeResponseProcessor"]
