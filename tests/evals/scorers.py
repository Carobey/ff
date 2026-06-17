"""Eval scorers: pure functions that return 0.0 (fail) or 1.0 (pass)."""

from __future__ import annotations

from typing import Any


def exact_match(actual: Any, expected: Any) -> float:
    """Return 1.0 iff actual == expected."""
    return 1.0 if actual == expected else 0.0


def threshold(value: float, op: str, threshold_val: float) -> float:
    """Return 1.0 iff the numeric condition holds."""
    ops: dict[str, bool] = {
        ">=": value >= threshold_val,
        "<=": value <= threshold_val,
        ">": value > threshold_val,
        "<": value < threshold_val,
        "==": value == threshold_val,
    }
    if op not in ops:
        raise ValueError(f"Unknown op: {op!r}. Allowed: {list(ops)}")
    return 1.0 if ops[op] else 0.0


def contains_any(text: Any, tokens: Any) -> float:
    """Return 1.0 iff ``text`` contains at least one of ``tokens`` (substring)."""
    haystack = str(text)
    return 1.0 if any(str(token) in haystack for token in (tokens or [])) else 0.0


def apply_scorer(
    scorer_cfg: dict[str, Any],
    result: dict[str, Any],
    expected: dict[str, Any],
) -> float:
    """Dispatch to the right scorer based on scorer_cfg['type']."""
    stype = scorer_cfg["type"]

    if stype == "exact_match":
        field = scorer_cfg["field"]
        return exact_match(result.get(field), expected.get(field))

    if stype == "contains_any":
        # Grounding check: the answer must echo a number/date that came from a tool.
        # ``expected[field]`` is the list of acceptable substrings (e.g. date variants).
        field = scorer_cfg["field"]
        return contains_any(result.get(field), expected.get(field))

    if stype == "threshold":
        field = scorer_cfg["field"]
        raw = result.get(field, 0.0)
        return threshold(float(raw), scorer_cfg["op"], float(scorer_cfg["value"]))

    if stype == "tool_call":
        # Tool-call correctness: did the agent pick the expected tool/route?
        field = scorer_cfg["field"]
        return exact_match(result.get(field), expected.get(field))

    raise ValueError(f"Unknown scorer type: {stype!r}")
