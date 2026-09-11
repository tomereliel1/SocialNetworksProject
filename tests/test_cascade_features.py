import math

from src.cascade_builder import build_interaction_graph
from src.cascade_features import extract_interaction_features
from src.cascade_validation import validate_cascade
from src.relation_parser import InteractionRelation


def graph_from_propagation_edges(edges):
    relations = [
        InteractionRelation(child, parent, f"user_{child}", f"user_{parent}")
        for parent, child in edges
    ]
    return build_interaction_graph(relations, interaction_type="retweet", direction="propagation")


def features_for(graph):
    validation = validate_cascade(graph)
    return extract_interaction_features(graph, prefix="rt", validation=validation)


def test_empty_graph_features_follow_nan_rules():
    graph = build_interaction_graph([], interaction_type="retweet", direction="propagation")
    features = features_for(graph)

    assert features["rt_n_nodes"] == 0
    assert features["rt_n_edges"] == 0
    assert features["rt_n_components"] == 0
    assert features["rt_n_users"] == 0
    assert math.isnan(features["rt_repeat_user_ratio"])
    assert math.isnan(features["rt_max_depth"])
    assert math.isnan(features["rt_structural_virality"])


def test_single_relation_features():
    graph = graph_from_propagation_edges([("A", "B")])
    features = features_for(graph)

    assert features["rt_n_nodes"] == 2
    assert features["rt_n_edges"] == 1
    assert features["rt_n_components"] == 1
    assert features["rt_n_roots"] == 1
    assert features["rt_n_leaves"] == 1
    assert features["rt_max_depth"] == 1
    assert features["rt_max_breadth"] == 1
    assert features["rt_avg_branching_factor"] == 1
    assert features["rt_structural_virality"] == 1


def test_chain_depth_breadth_and_virality():
    graph = graph_from_propagation_edges([("A", "B"), ("B", "C"), ("C", "D")])
    features = features_for(graph)

    assert features["rt_n_nodes"] == 4
    assert features["rt_n_components"] == 1
    assert features["rt_n_roots"] == 1
    assert features["rt_n_leaves"] == 1
    assert features["rt_max_depth"] == 3
    assert features["rt_max_breadth"] == 1


def test_chain_virality_is_larger_than_same_sized_star():
    chain = features_for(graph_from_propagation_edges([("A", "B"), ("B", "C"), ("C", "D")]))
    star = features_for(graph_from_propagation_edges([("A", "B"), ("A", "C"), ("A", "D")]))

    assert chain["rt_structural_virality"] > star["rt_structural_virality"]
    assert star["rt_max_depth"] == 1
    assert star["rt_max_breadth"] == 3


def test_disconnected_components_are_aggregated_to_one_event():
    graph = graph_from_propagation_edges([("A", "B"), ("B", "C"), ("X", "Y")])
    features = features_for(graph)

    assert features["rt_n_components"] == 2
    assert features["rt_largest_component_nodes"] == 3
    assert features["rt_largest_component_node_fraction"] == 3 / 5
    assert features["rt_max_depth"] == 2
    assert features["rt_largest_component_depth"] == 2
    assert features["rt_mean_component_depth"] == 1.5


def test_cyclic_component_keeps_defined_metrics_but_depth_is_nan():
    graph = graph_from_propagation_edges([("A", "B"), ("B", "A")])
    features = features_for(graph)

    assert features["rt_has_cycle"]
    assert features["rt_n_nodes"] == 2
    assert features["rt_n_edges"] == 2
    assert math.isnan(features["rt_max_depth"])
    assert math.isnan(features["rt_max_breadth"])
    assert features["rt_structural_virality"] == 1
