"""Utility helpers shared across the Azure Search MCP server modules."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, Optional, Sequence, Union, Callable, List, Iterable


def _coalesce(*values: Optional[Any]) -> Optional[Any]:
    """Return the first non-None value from the provided sequence."""

    for value in values:
        if value is not None:
            return value
    return None


def _normalize_sequence(
    value: Optional[Union[str, Sequence[Any]]],
    *,
    cast: Callable[[Any], Any],
) -> List[Any]:
    """Normalize strings, comma/newline separated values, or sequences into a list."""

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [cast(item) for item in value if item is not None]

    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, (list, tuple)):
                    return [cast(item) for item in parsed if item is not None]

            parts = re.split(r"[\n,]", stripped)
            return [cast(part.strip()) for part in parts if part and part.strip()]

        return []

    # Single primitive value
    return [cast(value)]


def _ensure_list_of_strings(value: Optional[Union[str, Sequence[Any]]]) -> List[str]:
    """Normalize value into list of strings with whitespace trimmed."""

    return _normalize_sequence(value, cast=lambda item: str(item).strip())


def _ensure_list_of_ints(value: Optional[Union[str, Sequence[Any]]]) -> List[int]:
    """Normalize value into list of integers."""

    return _normalize_sequence(value, cast=lambda item: int(str(item).strip()))


def _ensure_list_of_floats(value: Optional[Union[str, Sequence[Any]]]) -> List[float]:
    """Normalize value into list of floats."""

    return _normalize_sequence(value, cast=lambda item: float(str(item).strip()))


def _list_to_field_value(values: Sequence[str]) -> Optional[List[str]]:
    """Convert list of strings to the list representation expected by Azure SDK."""

    cleaned = [value for value in (value.strip() for value in values) if value]
    if not cleaned:
        return None

    return cleaned


def _vector_field_selector(values: Sequence[str]) -> str:
    """Render vector field value for the Azure SDK (comma-separated)."""

    cleaned = [value for value in (value.strip() for value in values) if value]
    if not cleaned:
        return "text_vector"

    return ",".join(cleaned)


def _normalize_vector_descriptors(
    value: Optional[Union[str, Sequence[Any]]]
) -> List[tuple[str, Optional[int], Optional[float]]]:
    """Normalize vector inputs into (text, k, weight) tuples."""

    descriptors: List[tuple[str, Optional[int], Optional[float]]] = []

    def _record(text: str, k: Optional[int] = None, weight: Optional[float] = None) -> None:
        stripped = text.strip()
        if stripped:
            descriptors.append((stripped, k, weight))

    def _parse_entry(entry: Any) -> None:
        if entry is None:
            return

        if isinstance(entry, str):
            stripped = entry.strip()
            if not stripped:
                return
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, (list, tuple)):
                    _parse_entry(parsed)
                    return
            _record(stripped)
            return

        if isinstance(entry, (list, tuple)):
            if not entry:
                return

            text = str(entry[0]).strip()
            if not text:
                return

            k_value: Optional[int] = None
            weight_value: Optional[float] = None

            if len(entry) > 1 and entry[1] not in (None, ""):
                try:
                    k_value = int(str(entry[1]).strip())
                except (TypeError, ValueError):
                    print(
                        f"Warning: unable to parse vector k from '{entry[1]}'",
                        file=sys.stderr,
                    )

            if len(entry) > 2 and entry[2] not in (None, ""):
                try:
                    weight_value = float(str(entry[2]).strip())
                except (TypeError, ValueError):
                    print(
                        f"Warning: unable to parse vector weight from '{entry[2]}'",
                        file=sys.stderr,
                    )

            _record(text, k_value, weight_value)
            return

        _record(str(entry))

    if value is None:
        return descriptors

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return descriptors

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, (list, tuple)):
                for item in parsed:
                    _parse_entry(item)
                return descriptors

        for line in (line.strip() for line in value.splitlines()):
            if line:
                _parse_entry(line)
        return descriptors

    if isinstance(value, (list, tuple, set)):
        for item in value:
            _parse_entry(item)
        return descriptors

    _parse_entry(value)
    return descriptors


def _parse_semantic_captions(value: str) -> tuple[dict[str, Any], bool]:
    """Translate REST-style captions string into SDK keyword arguments and return highlight flag."""

    if not value:
        return {}, False

    caption_type: Optional[str] = None
    highlight: Optional[bool] = None

    for part in (segment.strip() for segment in value.split("|") if segment.strip()):
        lowered = part.lower()
        if lowered.startswith("highlight-"):
            option = lowered.split("-", 1)[-1]
            highlight = option == "true"
        else:
            caption_type = part

    payload: dict[str, Any] = {}
    if caption_type:
        payload["query_caption"] = caption_type
    if highlight is not None:
        payload["query_caption_highlight_enabled"] = highlight

    return payload, bool(highlight)


def _parse_semantic_answers(value: str) -> dict[str, Any]:
    """Translate REST-style answers string into SDK keyword arguments."""

    if not value:
        return {}

    answer_type: Optional[str] = None
    answer_count: Optional[int] = None
    answer_threshold: Optional[float] = None

    for part in (segment.strip() for segment in value.split("|") if segment.strip()):
        lowered = part.lower()
        if lowered.startswith("count-"):
            try:
                answer_count = int(lowered.split("-", 1)[-1])
            except ValueError:
                print(f"Warning: unable to parse answer count from '{part}'", file=sys.stderr)
        elif lowered.startswith("threshold-"):
            try:
                answer_threshold = float(lowered.split("-", 1)[-1])
            except ValueError:
                print(f"Warning: unable to parse answer threshold from '{part}'", file=sys.stderr)
        else:
            answer_type = part

    payload: dict[str, Any] = {}
    if answer_type:
        payload["query_answer"] = answer_type
    if answer_count is not None:
        payload["query_answer_count"] = answer_count
    if answer_threshold is not None:
        payload["query_answer_threshold"] = answer_threshold

    return payload


def _try_parse_int(value: Optional[Union[str, int]]) -> Optional[int]:
    """Safely parse integers from environment values."""

    if value is None:
        return None

    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        print(f"Warning: unable to parse integer from value '{value}'", file=sys.stderr)
        return None


def _try_parse_float(value: Optional[Union[str, float]]) -> Optional[float]:
    """Safely parse floats from environment values."""

    if value is None:
        return None

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        print(f"Warning: unable to parse float from value '{value}'", file=sys.stderr)
        return None


__all__ = [name for name in globals() if name.startswith("_")]

