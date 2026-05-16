# -*- coding: utf-8 -*-
"""Tests for LiteLLM generation-parameter recovery."""

from src.llm.errors import (
    call_litellm_with_param_recovery,
    classify_litellm_generation_param_error,
)
from src.llm.generation_params import (
    apply_litellm_generation_params,
    clear_litellm_generation_param_recovery_cache,
)


def test_temperature_default_only_error_sets_temperature_to_one() -> None:
    recovery = classify_litellm_generation_param_error(
        RuntimeError(
            "Unsupported value: 'temperature' does not support 0.7 with this model. "
            "Only the default (1.0) value is supported."
        )
    )

    assert recovery is not None
    assert recovery.set_params == {"temperature": 1.0}
    assert recovery.omit_params == ()


def test_unsupported_temperature_error_retries_once_and_caches_recovery() -> None:
    clear_litellm_generation_param_recovery_cache()
    calls = []

    def _call(kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("Unsupported parameter: temperature is not supported")
        return "ok"

    result = call_litellm_with_param_recovery(
        _call,
        model="openai/custom-temp-locked",
        call_kwargs={
            "model": "openai/custom-temp-locked",
            "messages": [],
            "temperature": 0.7,
        },
    )
    future_kwargs = apply_litellm_generation_params(
        {"model": "openai/custom-temp-locked", "messages": []},
        "openai/custom-temp-locked",
        0.7,
    )

    assert result == "ok"
    assert calls[0]["temperature"] == 0.7
    assert "temperature" not in calls[1]
    assert "temperature" not in future_kwargs
