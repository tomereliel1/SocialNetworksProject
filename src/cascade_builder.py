from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import networkx as nx

from src.relation_parser import InteractionRelation


GraphDirection = Literal["action", "propagation"]


def build_interaction_graph(
    relations: Iterable[InteractionRelation],
    interaction_type: str,
    direction: GraphDirection = "propagation",
) -> nx.DiGraph:
    """Build a tweet-level graph from MC-Fake interaction relations.

    Args:
        relations: Parsed interaction relations.
        interaction_type: Relation family, for example `retweet` or `reply`.
        direction: `action` keeps A -> B, where A acts on B. `propagation`
            reverses that to B -> A, where the parent/source tweet points to
            the tweet that reacted to it.

    Returns:
        A NetworkX directed graph whose nodes are tweet IDs. Each node has a
        `user_id` attribute when present in a relation.

    Raises:
        ValueError: If `direction` is not `action` or `propagation`.
    """

    if direction not in {"action", "propagation"}:
        raise ValueError("direction must be either 'action' or 'propagation'")

    graph = nx.DiGraph(
        direction=direction,
        interaction_type=interaction_type,
        node_type="tweet_id",
    )
    seen_relations: set[InteractionRelation] = set()
    seen_edges: set[tuple[str, str]] = set()
    relation_count = 0
    duplicate_relations = 0
    duplicate_edges = 0

    for relation in relations:
        relation_count += 1
        duplicate_relations += int(relation in seen_relations)
        seen_relations.add(relation)

        graph.add_node(relation.tweet_a, user_id=relation.user_a)
        graph.add_node(relation.tweet_b, user_id=relation.user_b)

        source, target = _edge_nodes(relation, direction)
        duplicate_edges += int((source, target) in seen_edges)
        seen_edges.add((source, target))

        if graph.has_edge(source, target):
            graph[source][target]["relation_count"] += 1
        else:
            graph.add_edge(source, target, relation_count=1)

    graph.graph["n_relation_tokens"] = relation_count
    graph.graph["n_duplicate_relations"] = duplicate_relations
    graph.graph["n_duplicate_edges"] = duplicate_edges
    return graph


def _edge_nodes(relation: InteractionRelation, direction: GraphDirection) -> tuple[str, str]:
    """Return source and target tweet IDs for the requested direction.

    Args:
        relation: Parsed interaction relation.
        direction: Graph edge direction.

    Returns:
        `(source, target)` tweet IDs.
    """

    if direction == "action":
        return relation.tweet_a, relation.tweet_b
    return relation.tweet_b, relation.tweet_a


# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO:
# This user-level graph builder is currently unused in the repo. If it remains
# unused, remove it to keep cascade_builder focused on the tweet-level graphs
# that the validation pipeline actually consumes.
def build_user_interaction_graph(
    relations: Iterable[InteractionRelation],
    interaction_type: str,
    direction: GraphDirection = "propagation",
) -> nx.DiGraph:
    """Build a user-level graph for validation.

    Args:
        relations: Parsed interaction relations.
        interaction_type: Relation family, for example `retweet` or `reply`.
        direction: `action` keeps user A -> user B. `propagation` reverses to
            user B -> user A.

    Returns:
        A NetworkX directed graph whose nodes are user IDs.

    Raises:
        ValueError: If `direction` is not `action` or `propagation`.
    """

    if direction not in {"action", "propagation"}:
        raise ValueError("direction must be either 'action' or 'propagation'")

    graph = nx.DiGraph(
        direction=direction,
        interaction_type=interaction_type,
        node_type="user_id",
    )
    for relation in relations:
        if direction == "action":
            source, target = relation.user_a, relation.user_b
        else:
            source, target = relation.user_b, relation.user_a
        if graph.has_edge(source, target):
            graph[source][target]["relation_count"] += 1
        else:
            graph.add_edge(source, target, relation_count=1)
    return graph
