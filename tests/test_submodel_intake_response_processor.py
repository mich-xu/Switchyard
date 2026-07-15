# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for emitting routing-strategy model calls to intake.

The context is built the way an HTTP endpoint builds it — via
``attach_request_metadata`` — so these tests exercise the same session-id and
capture-opt-in path production uses, not a hand-seeded metadata shape.
"""

from switchyard.lib.config import IntakeSinkConfig
from switchyard.lib.processors.intake_payload_builder import CTX_SUBMODEL_CALLS
from switchyard.lib.processors.submodel_intake_response_processor import (
    SubModelIntakeResponseProcessor,
)
from switchyard.lib.proxy_context import ProxyContext
from switchyard.lib.request_metadata import (
    IntakeRequestMetadata,
    RequestMetadata,
    attach_request_metadata,
)

_RESPONSE = object()  # processor returns it unchanged; never inspected


class _FakeIntakeClient:
    """Captures the payloads the processor would POST, running the factory
    synchronously so tests can assert on the built payload."""

    def __init__(self) -> None:
        self.background_payloads: list[dict[str, object]] = []

    def enqueue_background(self, payload_factory):
        self.background_payloads.append(payload_factory())


def _config() -> IntakeSinkConfig:
    return IntakeSinkConfig(
        intake_base_url="http://localhost:8080",
        workspace="default",
        user_id="rlempka",
    )


def _classifier_call() -> dict[str, object]:
    return {
        "model": "gcp/google/gemini-3.5-flash",
        "prompt_tokens": 1200,
        "completion_tokens": 80,
        "cached_tokens": 0,
        "router_type": "deterministic",
        "routed_to": "classifier",
    }


def _ctx(*, opted_in: bool, calls: list[object] | None = None) -> ProxyContext:
    """Build a ProxyContext the way an endpoint does: RequestMetadata carries the
    session id + capture opt-in, exactly as ``attach_request_metadata`` sets it."""
    ctx = ProxyContext()
    attach_request_metadata(
        ctx,
        RequestMetadata(
            session_id="sess-123",
            intake=IntakeRequestMetadata(enabled=opted_in, task="my-task"),
        ),
    )
    ctx.metadata[CTX_SUBMODEL_CALLS] = [_classifier_call()] if calls is None else calls
    return ctx


async def test_emits_submodel_record_with_tokens_and_routing() -> None:
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)

    result = await processor.process(_ctx(opted_in=True), _RESPONSE)

    assert result is _RESPONSE
    assert len(client.background_payloads) == 1
    payload = client.background_payloads[0]
    assert payload["request"]["model"] == "gcp/google/gemini-3.5-flash"
    assert payload["request"]["switchyard"]["routing"] == {
        "router_type": "deterministic",
        "routed_to": "classifier",
    }
    assert payload["response"]["usage"]["prompt_tokens"] == 1200
    assert payload["response"]["usage"]["completion_tokens"] == 80
    assert payload["session_id"] == "sess-123"
    assert payload["user_id"] == "rlempka"
    # session id present -> the routing call joins the routed turn's eval run.
    assert payload["evaluation_context"]["evaluation_run_id"] == "sess-123"
    # created_at present so NVDataflow can time-bucket the record.
    assert payload["request"]["switchyard"]["created_at"]


async def test_record_carries_no_message_content() -> None:
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)

    await processor.process(_ctx(opted_in=True), _RESPONSE)

    request_entry = client.background_payloads[0]["request"]
    # Anonymous telemetry: only model + switchyard metadata, never the prompt.
    assert "messages" not in request_entry
    assert set(request_entry) == {"model", "switchyard"}


async def test_priced_submodel_call_has_positive_cost() -> None:
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)

    await processor.process(_ctx(opted_in=True), _RESPONSE)

    # gemini-3.5-flash is priced, so a real call must show non-zero cost.
    assert client.background_payloads[0]["cost_usd"] > 0


async def test_multiple_calls_emit_one_record_each() -> None:
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)
    ctx = _ctx(
        opted_in=True,
        calls=[
            _classifier_call(),
            {
                "model": "nvidia/nvidia/nemotron-3-super-v3",
                "prompt_tokens": 300,
                "completion_tokens": 20,
                "cached_tokens": 0,
                "router_type": "plan_execute",
                "routed_to": "planner",
            },
        ],
    )

    await processor.process(ctx, _RESPONSE)

    assert len(client.background_payloads) == 2
    routed = {p["request"]["switchyard"]["routing"]["routed_to"] for p in client.background_payloads}
    assert routed == {"classifier", "planner"}


async def test_not_opted_in_emits_nothing() -> None:
    """A turn that did not opt into capture must not leak a routing record."""
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)

    await processor.process(_ctx(opted_in=False), _RESPONSE)

    assert client.background_payloads == []


async def test_no_request_metadata_emits_nothing() -> None:
    """No RequestMetadata (never opted in) is treated as opt-out, not a leak."""
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)
    ctx = ProxyContext()
    ctx.metadata[CTX_SUBMODEL_CALLS] = [_classifier_call()]

    await processor.process(ctx, _RESPONSE)

    assert client.background_payloads == []


async def test_no_submodel_calls_emits_nothing() -> None:
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)

    await processor.process(_ctx(opted_in=True, calls=[]), _RESPONSE)

    assert client.background_payloads == []


async def test_malformed_entries_are_skipped() -> None:
    """Non-dict and invalid entries are skipped — no synthetic zero-cost record."""
    client = _FakeIntakeClient()
    processor = SubModelIntakeResponseProcessor(_config(), client=client)
    ctx = _ctx(
        opted_in=True,
        calls=[
            None,
            "oops",
            {},  # no model
            {"model": "", "prompt_tokens": 1, "completion_tokens": 1},  # empty model
            {"model": "m", "prompt_tokens": -5, "completion_tokens": 1},  # negative tokens
            _classifier_call(),  # the one valid entry
        ],
    )

    await processor.process(ctx, _RESPONSE)

    assert len(client.background_payloads) == 1
    assert client.background_payloads[0]["request"]["model"] == "gcp/google/gemini-3.5-flash"


async def test_shutdown_does_not_close_injected_client() -> None:
    """Ownership boundary: shutdown() never closes a client it did not build.

    ``_FakeIntakeClient`` has no ``aclose``; if the ownership guard regressed and
    shutdown() tried to close the injected client, this would raise.
    """
    processor = SubModelIntakeResponseProcessor(_config(), client=_FakeIntakeClient())

    await processor.shutdown()
