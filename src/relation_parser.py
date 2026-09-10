from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any


RELATION_DELIMITER = ","
ID_DELIMITER = "-"


@dataclass(frozen=True)
class InteractionRelation:
    """One MC-Fake tweet-to-tweet interaction relation.

    Attributes:
        tweet_a: The tweet that performs the interaction.
        tweet_b: The tweet being interacted with.
        user_a: The author/user ID of `tweet_a`.
        user_b: The author/user ID of `tweet_b`.
    """

    tweet_a: str
    tweet_b: str
    user_a: str
    user_b: str


@dataclass(frozen=True)
class MalformedRelation:
    """A relation token that could not be parsed.

    Attributes:
        token: The raw token text after basic cleanup.
        reason: Human-readable explanation of why parsing failed.
        token_index: Zero-based position of the token inside the raw cell.
    """

    token: str
    reason: str
    token_index: int


@dataclass(frozen=True)
class RelationParseResult:
    """Parsed relation data from one CSV cell.

    Attributes:
        relations: Valid parsed relation tokens.
        malformed: Malformed tokens found in permissive mode.
        duplicate_relations: Number of repeated valid relation tokens.
        is_empty: Whether the raw cell was missing or empty.
    """

    relations: list[InteractionRelation]
    malformed: list[MalformedRelation]
    duplicate_relations: int
    is_empty: bool


class RelationParseError(ValueError):
    """Raised when strict parsing encounters a malformed relation token."""


def _is_missing(raw_value: Any) -> bool:
    """Return whether a raw pandas cell value should be treated as missing.

    Args:
        raw_value: A raw value from pandas or a hand-written test value.

    Returns:
        True when the value is null-like.
    """

    if raw_value is None:
        return True
    if isinstance(raw_value, float):
        return isnan(raw_value)
    try:
        import pandas as pd

        return bool(pd.isna(raw_value))
    except (ImportError, TypeError, ValueError):
        return False


def _stringify_raw_value(raw_value: Any) -> str:
    """Convert a raw cell value to text without numeric coercion.

    Args:
        raw_value: Raw CSV cell value.

    Returns:
        Text representation preserving large IDs when the input is already a
        string.
    """

    if isinstance(raw_value, str):
        return raw_value
    return str(raw_value)


def parse_interaction_relations(raw_value: Any, strict: bool = False) -> RelationParseResult:
    """Parse a raw MC-Fake relation cell.

    The observed MC-Fake serialization for both retweets and replies is a
    comma-separated list of tokens. Each token contains four hyphen-separated
    numeric IDs:

    `tweet_A-tweet_B-user_A-user_B`

    For retweets, A retweets B. For replies, A replies B.

    Args:
        raw_value: Raw cell value as returned by pandas.
        strict: If True, raise `RelationParseError` when malformed tokens are
            found. If False, keep valid tokens and report malformed tokens.

    Returns:
        A `RelationParseResult` containing valid relations and diagnostics.

    Raises:
        RelationParseError: If `strict` is True and a malformed token exists.
    """

    if _is_missing(raw_value):
        return RelationParseResult([], [], 0, True)

    raw_text = _stringify_raw_value(raw_value).strip()
    if raw_text == "" or raw_text.lower() in {"nan", "none", "null"}:
        return RelationParseResult([], [], 0, True)

    relations: list[InteractionRelation] = []
    malformed: list[MalformedRelation] = []

    for token_index, token in enumerate(raw_text.split(RELATION_DELIMITER)):
        clean_token = token.strip().strip('"').strip("'").strip()
        if clean_token == "":
            malformed.append(MalformedRelation(token, "empty relation token", token_index))
            continue

        parts = [part.strip() for part in clean_token.split(ID_DELIMITER)]
        if len(parts) != 4:
            malformed.append(
                MalformedRelation(
                    clean_token,
                    f"expected 4 hyphen-separated IDs, found {len(parts)}",
                    token_index,
                )
            )
            continue

        if any(part == "" for part in parts):
            malformed.append(MalformedRelation(clean_token, "one or more IDs are empty", token_index))
            continue

        if any(not part.isdigit() for part in parts):
            malformed.append(MalformedRelation(clean_token, "one or more IDs are not numeric", token_index))
            continue

        relations.append(InteractionRelation(*parts))

    duplicate_relations = len(relations) - len(set(relations))

    if strict and malformed:
        first = malformed[0]
        raise RelationParseError(
            f"Malformed relation at token {first.token_index}: {first.reason}: {first.token}"
        )

    return RelationParseResult(relations, malformed, duplicate_relations, False)
