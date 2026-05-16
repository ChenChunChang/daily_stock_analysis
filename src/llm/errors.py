# -*- coding: utf-8 -*-
"""LiteLLM error classification and one-shot parameter recovery."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.llm.generation_params import (
    GenerationParamRecovery,
    apply_litellm_param_recovery,
    remember_litellm_generation_param_recovery,
)

_UNSUPPORTED_PARAM_MARKERS = (
    "unsupported",
    "not supported",
    "unrecognized",
    "unknown parameter",
    "not allowed",
    "invalid parameter",
    "does not support",
)


def _collect_error_text(value: Any, seen: Optional[set] = None) -> List[str]:
    if seen is None:
        seen = set()
    if value is None:
        return []
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)

    chunks = [str(value)]
    if isinstance(value, BaseException):
        chunks.extend(_collect_error_text(getattr(value, "args", None), seen))
    if isinstance(value, dict):
        for item in value.values():
            chunks.extend(_collect_error_text(item, seen))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            chunks.extend(_collect_error_text(item, seen))
    else:
        for attr in ("message", "body", "response", "llm_provider", "param"):
            if hasattr(value, attr):
                chunks.extend(_collect_error_text(getattr(value, attr), seen))
    return chunks


def _normalized_error_text(error: BaseException) -> str:
    return " ".join(chunk for chunk in _collect_error_text(error) if chunk).lower()


def classify_litellm_generation_param_error(
    error: BaseException,
) -> Optional[GenerationParamRecovery]:
    """Classify explicit provider parameter errors into a safe one-shot recovery."""
    text = _normalized_error_text(error)
    if not text:
        return None

    if "temperature" in text:
        if "only" in text and ("1.0" in text or "default" in text or " value 1" in text):
            return GenerationParamRecovery(
                set_params={"temperature": 1.0},
                reason="temperature_default_only",
            )
        if any(marker in text for marker in _UNSUPPORTED_PARAM_MARKERS):
            return GenerationParamRecovery(
                omit_params=("temperature",),
                reason="temperature_unsupported",
            )

    for param in ("top_p", "presence_penalty", "frequency_penalty", "seed"):
        if param in text and any(marker in text for marker in _UNSUPPORTED_PARAM_MARKERS):
            return GenerationParamRecovery(
                omit_params=(param,),
                reason=f"{param}_unsupported",
            )
    return None


def call_litellm_with_param_recovery(
    call: Callable[[Dict[str, Any]], Any],
    *,
    model: str,
    call_kwargs: Dict[str, Any],
    model_list: Optional[List[Dict[str, Any]]] = None,
    logger: Optional[Any] = None,
    log_label: str = "[LiteLLM]",
) -> Any:
    """Call LiteLLM once, then retry once for explicit generation-parameter errors."""
    effective_kwargs = dict(call_kwargs)
    try:
        return call(effective_kwargs)
    except Exception as exc:
        recovery = classify_litellm_generation_param_error(exc)
        if recovery is None:
            raise
        retry_kwargs = apply_litellm_param_recovery(effective_kwargs, recovery)
        if retry_kwargs == effective_kwargs:
            raise
        if logger is not None:
            logger.warning(
                "%s %s generation parameter rejected (%s), retrying once with request-scoped recovery",
                log_label,
                model,
                recovery.reason,
            )
        response = call(retry_kwargs)
        remember_litellm_generation_param_recovery(
            model,
            recovery,
            model_list=model_list,
            request_overrides=retry_kwargs,
        )
        return response
