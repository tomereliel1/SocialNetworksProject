import math

import numpy as np
import pytest

from src.relation_parser import (
    InteractionRelation,
    RelationParseError,
    parse_interaction_relations,
)


REAL_MC_FAKE_EXAMPLE = (
    "1258266492573044736-1258265925658259458-722618840371372037-284967242,"
    "1258268929732104192-1258265925658259458-2309085185-284967242"
)


def test_one_valid_relation():
    result = parse_interaction_relations("1-2-3-4")
    assert result.relations == [InteractionRelation("1", "2", "3", "4")]
    assert result.malformed == []


def test_multiple_valid_relations_comma_delimited():
    result = parse_interaction_relations("1-2-3-4,5-6-7-8")
    assert result.relations == [
        InteractionRelation("1", "2", "3", "4"),
        InteractionRelation("5", "6", "7", "8"),
    ]


def test_empty_string():
    result = parse_interaction_relations("")
    assert result.is_empty
    assert result.relations == []


@pytest.mark.parametrize("missing", [None, math.nan, np.nan])
def test_missing_values(missing):
    result = parse_interaction_relations(missing)
    assert result.is_empty
    assert result.relations == []


def test_whitespace_is_stripped():
    result = parse_interaction_relations("  1 - 2 - 3 - 4 , 5-6-7-8  ")
    assert result.relations[0] == InteractionRelation("1", "2", "3", "4")
    assert result.relations[1] == InteractionRelation("5", "6", "7", "8")


def test_malformed_token_with_too_few_ids():
    result = parse_interaction_relations("1-2-3")
    assert result.relations == []
    assert result.malformed[0].reason == "expected 4 hyphen-separated IDs, found 3"


def test_malformed_token_with_too_many_ids():
    result = parse_interaction_relations("1-2-3-4-5")
    assert result.relations == []
    assert result.malformed[0].reason == "expected 4 hyphen-separated IDs, found 5"


def test_strict_mode_raises_on_malformed_token():
    with pytest.raises(RelationParseError):
        parse_interaction_relations("1-2-3", strict=True)


def test_duplicate_relations_are_counted():
    result = parse_interaction_relations("1-2-3-4,1-2-3-4")
    assert len(result.relations) == 2
    assert result.duplicate_relations == 1


def test_large_twitter_ids_remain_strings():
    result = parse_interaction_relations(
        "1258266492573044736-1258265925658259458-722618840371372037-284967242"
    )
    assert result.relations[0].tweet_a == "1258266492573044736"


def test_real_mc_fake_raw_example():
    result = parse_interaction_relations(REAL_MC_FAKE_EXAMPLE)
    assert len(result.relations) == 2
    assert result.relations[0] == InteractionRelation(
        "1258266492573044736",
        "1258265925658259458",
        "722618840371372037",
        "284967242",
    )


def test_reply_relations_use_same_real_world_format():
    raw_reply = "1258265928321699840-1258265925658259458-284967242-284967242"
    result = parse_interaction_relations(raw_reply)
    assert result.relations == [
        InteractionRelation(
            "1258265928321699840",
            "1258265925658259458",
            "284967242",
            "284967242",
        )
    ]
