from src.cascade_builder import build_interaction_graph
from src.cascade_validation import (
    CONNECTED_CYCLIC,
    CONNECTED_TREE,
    DISCONNECTED_FOREST,
    EMPTY,
    MALFORMED,
    SINGLE_RELATION,
    validate_cascade,
)
from src.relation_parser import InteractionRelation


def test_empty_graph_status():
    graph = build_interaction_graph([], interaction_type="retweet")
    validation = validate_cascade(graph)
    assert validation.graph_status == EMPTY


def test_single_relation_status():
    graph = build_interaction_graph([InteractionRelation("a", "b", "ua", "ub")], interaction_type="retweet")
    validation = validate_cascade(graph)
    assert validation.graph_status == SINGLE_RELATION
    assert validation.n_roots == 1


def test_connected_tree_status():
    graph = build_interaction_graph(
        [
            InteractionRelation("b", "a", "ub", "ua"),
            InteractionRelation("c", "b", "uc", "ub"),
        ],
        interaction_type="retweet",
        direction="propagation",
    )
    validation = validate_cascade(graph)
    assert validation.graph_status == CONNECTED_TREE
    assert validation.is_tree


def test_disconnected_forest_status():
    graph = build_interaction_graph(
        [
            InteractionRelation("b", "a", "ub", "ua"),
            InteractionRelation("d", "c", "ud", "uc"),
        ],
        interaction_type="retweet",
        direction="propagation",
    )
    validation = validate_cascade(graph)
    assert validation.graph_status == DISCONNECTED_FOREST
    assert validation.n_weak_components == 2


def test_cycle_status():
    graph = build_interaction_graph(
        [
            InteractionRelation("b", "a", "ub", "ua"),
            InteractionRelation("a", "b", "ua", "ub"),
        ],
        interaction_type="retweet",
        direction="propagation",
    )
    validation = validate_cascade(graph)
    assert validation.graph_status == CONNECTED_CYCLIC
    assert validation.has_cycle


def test_parse_errors_make_status_malformed():
    graph = build_interaction_graph([InteractionRelation("a", "b", "ua", "ub")], interaction_type="retweet")
    validation = validate_cascade(graph, parse_error_count=1)
    assert validation.graph_status == MALFORMED
