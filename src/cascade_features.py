from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean, median, pstdev

import networkx as nx

from src.cascade_validation import (
    CONNECTED_DAG,
    CONNECTED_TREE,
    DISCONNECTED_FOREST,
    SINGLE_RELATION,
    CascadeValidation,
    validate_cascade,
)
from src.relation_parser import RelationParseResult


CLEAN_STRUCTURAL_STATUSES = {
    SINGLE_RELATION,
    CONNECTED_TREE,
    DISCONNECTED_FOREST,
    CONNECTED_DAG,
}


@dataclass(frozen=True)
class ComponentMetrics:
    """Structural metrics for one weakly connected component.

    Attributes:
        nodes: Number of nodes in the component.
        edges: Number of directed edges in the component.
        depth: Maximum root-based depth, or NaN when undefined.
        max_breadth: Largest number of nodes at any root-based layer, or NaN
            when undefined.
        virality: Average pairwise distance in the undirected component, or NaN
            when undefined.
    """

    nodes: int
    edges: int
    depth: float
    max_breadth: float
    virality: float


def extract_interaction_features(
    graph: nx.DiGraph,
    *,
    prefix: str,
    validation: CascadeValidation | None = None,
    parse_result: RelationParseResult | None = None,
) -> dict[str, object]:
    """Extract flat event-level structural features from one interaction graph.

    The graph must already use propagation direction, so roots are nodes with
    zero in-degree and leaves are nodes with zero out-degree. Disconnected weak
    components are measured independently and then aggregated back to one event
    row. Virality is computed on each component's undirected projection; status
    columns are kept so downstream analysis can restrict to clean trees/forests.

    Args:
        graph: Directed tweet-level interaction graph.
        prefix: Prefix to add to every output feature name, such as `rt` or
            `reply`.
        validation: Optional precomputed structural diagnostics for `graph`.
            When omitted, validation is computed inside this function.
        parse_result: Optional parser output for the source relation cell. When
            provided, parse diagnostics are included in the output features.

    Returns:
        A flat dictionary of prefixed feature names and CSV-friendly values.
    """

    validation = validation or validate_cascade(
        graph,
        parse_error_count=len(parse_result.malformed) if parse_result is not None else 0,
        duplicate_relations=parse_result.duplicate_relations if parse_result is not None else None,
    )

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    components = weak_components_sorted(graph)
    largest_component = largest_weak_component(components)
    component_metrics = [compute_component_metrics(graph.subgraph(nodes).copy()) for nodes in components]
    largest_metrics = (
        compute_component_metrics(graph.subgraph(largest_component).copy())
        if largest_component is not None
        else ComponentMetrics(0, 0, math.nan, math.nan, math.nan)
    )
    n_users = count_unique_users(graph)

    component_node_counts = [len(nodes) for nodes in components]
    largest_component_nodes = largest_metrics.nodes
    largest_component_edges = largest_metrics.edges

    root_count = sum(1 for _, degree in graph.in_degree() if degree == 0)
    leaf_count = sum(1 for _, degree in graph.out_degree() if degree == 0)
    positive_out_degrees = [degree for _, degree in graph.out_degree() if degree > 0]

    valid_depths = finite_values(metric.depth for metric in component_metrics)
    valid_breadths = finite_values(metric.max_breadth for metric in component_metrics)
    valid_viralities = finite_values(metric.virality for metric in component_metrics)

    return add_prefix(
        prefix,
        {
            "parse_malformed_count": parse_malformed_count(parse_result, validation),
            "parse_duplicate_count": parse_duplicate_count(parse_result, validation),
            "parse_is_empty": parse_result.is_empty if parse_result is not None else n_nodes == 0,
            "graph_status": validation.graph_status,
            "has_cycle": validation.has_cycle,
            "is_dag": validation.is_dag,
            "is_structurally_clean": validation.graph_status in CLEAN_STRUCTURAL_STATUSES,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_users": n_users,
            "repeat_user_ratio": math.nan if n_nodes == 0 else 1 - n_users / n_nodes,
            "n_components": len(components),
            "largest_component_nodes": largest_component_nodes,
            "largest_component_edges": largest_component_edges,
            "largest_component_node_fraction": (
                math.nan if n_nodes == 0 else largest_component_nodes / n_nodes
            ),
            "mean_component_nodes": math.nan if not component_node_counts else mean(component_node_counts),
            "median_component_nodes": math.nan if not component_node_counts else median(component_node_counts),
            "std_component_nodes": math.nan if not component_node_counts else pstdev(component_node_counts),
            "n_roots": root_count,
            "root_ratio": math.nan if n_nodes == 0 else root_count / n_nodes,
            "n_leaves": leaf_count,
            "leaves_ratio": math.nan if n_nodes == 0 else leaf_count / n_nodes,
            "avg_branching_factor": (
                math.nan if n_nodes == 0 else mean(positive_out_degrees) if positive_out_degrees else 0
            ),
            "max_out_degree": max((degree for _, degree in graph.out_degree()), default=0),
            "max_depth": max(valid_depths) if valid_depths else math.nan,
            "mean_component_depth": mean(valid_depths) if valid_depths else math.nan,
            "largest_component_depth": largest_metrics.depth,
            "max_breadth": max(valid_breadths) if valid_breadths else math.nan,
            "mean_component_max_breadth": mean(valid_breadths) if valid_breadths else math.nan,
            "largest_component_max_breadth": largest_metrics.max_breadth,
            "structural_virality": pair_weighted_virality(component_metrics),
            "mean_component_virality": mean(valid_viralities) if valid_viralities else math.nan,
            "largest_component_virality": largest_metrics.virality,
        },
    )


def weak_components_sorted(graph: nx.DiGraph) -> list[frozenset[str]]:
    """Return weak components in deterministic order.

    Args:
        graph: Directed graph whose weak components should be listed.

    Returns:
        Weakly connected components sorted by `component_sort_key`.
    """

    return sorted(
        (frozenset(component) for component in nx.weakly_connected_components(graph)),
        key=component_sort_key,
    )


def largest_weak_component(components: list[frozenset[str]]) -> frozenset[str] | None:
    """Return the largest component, with deterministic tie-breaking.

    Args:
        components: Weakly connected components.

    Returns:
        The component with the most nodes, or None when `components` is empty.
    """

    if not components:
        return None
    return min(components, key=lambda component: (-len(component), tuple(sorted(component))))


def compute_component_metrics(component: nx.DiGraph) -> ComponentMetrics:
    """Compute metrics that are meaningful only within one weak component.

    Args:
        component: Directed subgraph for one weakly connected component.

    Returns:
        Structural metrics for `component`.
    """

    depth, max_breadth = compute_component_depth_and_breadth(component)
    return ComponentMetrics(
        nodes=component.number_of_nodes(),
        edges=component.number_of_edges(),
        depth=depth,
        max_breadth=max_breadth,
        virality=compute_component_virality(component),
    )


def compute_component_depth_and_breadth(component: nx.DiGraph) -> tuple[float, float]:
    """Return root-based depth and breadth for an acyclic component.

    For DAGs with multiple roots, each node is assigned to its nearest root
    layer. Cyclic components return NaN because cascade depth is not well
    defined without changing the graph.

    Args:
        component: Directed graph for one weakly connected component.

    Returns:
        A `(depth, max_breadth)` pair. Values are NaN when the component is
        empty, cyclic, rootless, or not fully reachable from its roots.
    """

    if component.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(component):
        return math.nan, math.nan

    roots = [node for node, degree in component.in_degree() if degree == 0]
    if not roots:
        return math.nan, math.nan

    levels: dict[str, int] = {}
    for root in sorted(roots):
        for node, distance in nx.single_source_shortest_path_length(component, root).items():
            levels[node] = min(levels.get(node, distance), distance)

    if len(levels) != component.number_of_nodes():
        return math.nan, math.nan

    layer_counts: dict[int, int] = {}
    for level in levels.values():
        layer_counts[level] = layer_counts.get(level, 0) + 1

    return max(levels.values()), max(layer_counts.values())


def compute_component_virality(component: nx.DiGraph) -> float:
    """Return average pairwise distance within one weak component.

    Args:
        component: Directed graph for one weakly connected component.

    Returns:
        Average shortest-path length on the undirected projection, or NaN when
        the component has fewer than two nodes or is disconnected.
    """

    if component.number_of_nodes() < 2:
        return math.nan

    undirected = component.to_undirected()
    if not nx.is_connected(undirected):
        return math.nan
    return nx.average_shortest_path_length(undirected)


def pair_weighted_virality(component_metrics: list[ComponentMetrics]) -> float:
    """Aggregate component virality by weighting each component by node pairs.

    Args:
        component_metrics: Metrics for all weak components in one event graph.

    Returns:
        Pair-weighted virality across finite component viralities, or NaN when
        no component contributes a valid node pair.
    """

    weighted_sum = 0.0
    pair_count = 0
    for metric in component_metrics:
        if not math.isfinite(metric.virality):
            continue
        pairs = metric.nodes * (metric.nodes - 1) // 2
        weighted_sum += pairs * metric.virality
        pair_count += pairs

    return weighted_sum / pair_count if pair_count else math.nan


def count_unique_users(graph: nx.DiGraph) -> int:
    """Count distinct non-missing user IDs stored on graph nodes.

    Args:
        graph: Directed interaction graph with optional `user_id` node
            attributes.

    Returns:
        Number of distinct non-null user IDs.
    """

    return len(
        {
            graph.nodes[node].get("user_id")
            for node in graph.nodes
            if graph.nodes[node].get("user_id") is not None
        }
    )


def parse_malformed_count(
    parse_result: RelationParseResult | None,
    validation: CascadeValidation,
) -> int:
    """Return the parse malformed-token count from parser or validation.

    Args:
        parse_result: Optional parser output for the source relation cell.
        validation: Structural diagnostics for the graph.

    Returns:
        Malformed-token count from `parse_result` when available, otherwise the
        count stored in `validation`.
    """

    return len(parse_result.malformed) if parse_result is not None else validation.parse_error_count


def parse_duplicate_count(
    parse_result: RelationParseResult | None,
    validation: CascadeValidation,
) -> int:
    """Return the duplicate raw relation count from parser or validation.

    Args:
        parse_result: Optional parser output for the source relation cell.
        validation: Structural diagnostics for the graph.

    Returns:
        Duplicate raw relation count from `parse_result` when available,
        otherwise the count stored in `validation`.
    """

    return parse_result.duplicate_relations if parse_result is not None else validation.n_duplicate_relations


def finite_values(values: Iterable[float]) -> list[float]:
    """Return finite metric values from an iterable.

    Args:
        values: Numeric values that may include NaN or infinity.

    Returns:
        Values converted to float with non-finite values removed.
    """

    return [float(value) for value in values if math.isfinite(float(value))]


def component_sort_key(component: frozenset[str]) -> tuple[int, tuple[str, ...]]:
    """Sort smaller lexical component signatures first.

    Args:
        component: Component node IDs.

    Returns:
        A stable sort key ordered by component size and lexical node IDs.
    """

    return (len(component), tuple(sorted(component)))


def add_prefix(prefix: str, features: dict[str, object]) -> dict[str, object]:
    """Prefix every feature key for one interaction type.

    Args:
        prefix: Prefix to prepend to each feature name.
        features: Unprefixed feature names and values.

    Returns:
        A new dictionary with keys formatted as `{prefix}_{feature_name}`.
    """

    return {f"{prefix}_{key}": value for key, value in features.items()}
