from __future__ import annotations

from dataclasses import asdict, dataclass

import networkx as nx


EMPTY = "EMPTY"
SINGLE_RELATION = "SINGLE_RELATION"
CONNECTED_TREE = "CONNECTED_TREE"
CONNECTED_DAG = "CONNECTED_DAG"
CONNECTED_CYCLIC = "CONNECTED_CYCLIC"
DISCONNECTED_FOREST = "DISCONNECTED_FOREST"
DISCONNECTED_GENERAL = "DISCONNECTED_GENERAL"
MALFORMED = "MALFORMED"


@dataclass(frozen=True)
class CascadeValidation:
    """Structural validation results for one interaction graph.

    Attributes:
        n_nodes: Number of tweet nodes.
        n_edges: Number of unique directed graph edges.
        n_weak_components: Number of weakly connected components.
        n_strong_components: Number of strongly connected components.
        n_roots: Number of nodes with in-degree zero.
        n_leaves: Number of nodes with out-degree zero.
        has_cycle: Whether the directed graph contains a cycle.
        is_dag: Whether the directed graph is acyclic.
        is_weakly_connected: Whether all nodes belong to one weak component.
        is_tree: Whether the graph is one connected acyclic tree.
        n_self_loops: Number of self-loop edges.
        n_duplicate_relations: Number of repeated raw relation tokens.
        n_duplicate_edges: Number of repeated graph edges after direction is applied.
        parse_error_count: Number of malformed relation tokens.
        graph_status: Human-readable structural category.
    """

    n_nodes: int
    n_edges: int
    n_weak_components: int
    n_strong_components: int
    n_roots: int
    n_leaves: int
    has_cycle: bool
    is_dag: bool
    is_weakly_connected: bool
    is_tree: bool
    n_self_loops: int
    n_duplicate_relations: int
    n_duplicate_edges: int
    parse_error_count: int
    graph_status: str

    def to_dict(self) -> dict[str, object]:
        """Convert validation results to a CSV-friendly dictionary.

        Returns:
            Field names and values from this dataclass.
        """

        return asdict(self)


def classify_graph(
    graph: nx.DiGraph,
    has_cycle: bool,
    is_tree: bool,
    parse_error_count: int,
) -> str:
    """Assign one structural status to an interaction graph.

    Args:
        graph: Directed interaction graph.
        has_cycle: Whether `graph` contains a directed cycle.
        is_tree: Whether `graph` is one connected acyclic tree.
        parse_error_count: Number of malformed relation tokens in the source row.

    Returns:
        One of the graph status constants in this module.
    """

    if parse_error_count > 0:
        return MALFORMED
    if graph.number_of_edges() == 0:
        return EMPTY
    if graph.number_of_edges() == 1:
        return SINGLE_RELATION

    weak_components = [set(component) for component in nx.weakly_connected_components(graph)]
    if len(weak_components) == 1:
        if has_cycle:
            return CONNECTED_CYCLIC
        if is_tree:
            return CONNECTED_TREE
        return CONNECTED_DAG

    if not has_cycle and all(_component_is_tree(graph, component) for component in weak_components):
        return DISCONNECTED_FOREST
    return DISCONNECTED_GENERAL


def validate_cascade(
    graph: nx.DiGraph,
    parse_error_count: int = 0,
    duplicate_relations: int | None = None,
    duplicate_edges: int | None = None,
) -> CascadeValidation:
    """Calculate structural diagnostics for one interaction graph.

    Args:
        graph: Directed tweet-level interaction graph.
        parse_error_count: Number of malformed raw tokens in the source row.
        duplicate_relations: Optional override for duplicate raw token count.
        duplicate_edges: Optional override for duplicate edge count.

    Returns:
        A `CascadeValidation` object with counts and structural status.
    """

    duplicate_relations_count = _metadata_or_override(graph, "n_duplicate_relations", duplicate_relations)
    duplicate_edges_count = _metadata_or_override(graph, "n_duplicate_edges", duplicate_edges)

    if graph.number_of_nodes() == 0:
        return CascadeValidation(
            n_nodes=0,
            n_edges=0,
            n_weak_components=0,
            n_strong_components=0,
            n_roots=0,
            n_leaves=0,
            has_cycle=False,
            is_dag=True,
            is_weakly_connected=False,
            is_tree=False,
            n_self_loops=0,
            n_duplicate_relations=duplicate_relations_count,
            n_duplicate_edges=duplicate_edges_count,
            parse_error_count=parse_error_count,
            graph_status=MALFORMED if parse_error_count else EMPTY,
        )

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_weak_components = nx.number_weakly_connected_components(graph)
    is_dag = nx.is_directed_acyclic_graph(graph)
    is_tree = n_weak_components == 1 and is_dag and n_edges == n_nodes - 1 and n_edges > 0

    return CascadeValidation(
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_weak_components=n_weak_components,
        n_strong_components=nx.number_strongly_connected_components(graph),
        n_roots=sum(1 for _, degree in graph.in_degree() if degree == 0),
        n_leaves=sum(1 for _, degree in graph.out_degree() if degree == 0),
        has_cycle=not is_dag,
        is_dag=is_dag,
        is_weakly_connected=n_weak_components == 1,
        is_tree=is_tree,
        n_self_loops=nx.number_of_selfloops(graph),
        n_duplicate_relations=duplicate_relations_count,
        n_duplicate_edges=duplicate_edges_count,
        parse_error_count=parse_error_count,
        graph_status=classify_graph(graph, not is_dag, is_tree, parse_error_count),
    )


def is_strictly_usable(validation: CascadeValidation) -> bool:
    """Return whether an event-level graph is clean enough for strict metrics.

    Args:
        validation: Structural diagnostics for one event graph.

    Returns:
        True when the event graph is connected, rooted, acyclic, non-empty, and
        parse-error-free.
    """

    return (
        validation.parse_error_count == 0
        and validation.n_nodes >= 2
        and validation.n_edges >= 1
        and validation.is_weakly_connected
        and validation.is_dag
        and validation.n_roots == 1
    )


def _component_is_tree(graph: nx.DiGraph, nodes: set[str]) -> bool:
    """Return whether one weak component is an acyclic tree."""

    subgraph = graph.subgraph(nodes)
    expected_edges = max(0, subgraph.number_of_nodes() - 1)
    return subgraph.number_of_edges() == expected_edges and nx.is_directed_acyclic_graph(subgraph)


def _metadata_or_override(graph: nx.DiGraph, key: str, override: int | None) -> int:
    """Return an explicit value when provided, otherwise graph metadata."""

    return int(graph.graph.get(key, 0)) if override is None else override
