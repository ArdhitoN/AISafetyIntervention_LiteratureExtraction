from __future__ import annotations

from .merge_types import (
    NodeComparisonInput,
    NodeViewForComparison,
    NodeAggregate,
    NeighborNodeSummary,
    NodeSetComparisonInput,
)
from .merge_indexer import MergeIndex


def _node_view(agg: NodeAggregate, index: MergeIndex) -> NodeViewForComparison:
    contexts: list[str] = []
    for e in agg.linked_edges:
        if e.rationale:
            contexts.append(e.get_context_for_node(agg.node_key))

    # Build neighbor summaries (nodes only, no edges)
    neighbors: list[NeighborNodeSummary] = []
    for nk in agg.neighbor_node_keys:
        n = index.nodes.get(nk)
        if not n:
            continue
        # Lightweight neighbor context: neighbor notes + its linked edge rationales
        neighbor_contexts: list[str] = []
        for ne in n.linked_edges:
            if ne.rationale:
                neighbor_contexts.append(ne.get_context_for_node(n.node_key))
        if n.notes:
            neighbor_contexts.extend(n.notes)
        neighbors.append(
            NeighborNodeSummary(
                node_key=n.node_key,
                text=n.merged_text or n.canonical_text,
                aliases=n.aliases,
                context=neighbor_contexts,
            )
        )

    return NodeViewForComparison(
        merged_text=agg.merged_text or agg.canonical_text,
        aliases=agg.aliases,
        context=contexts + agg.notes,
        linked_edges=agg.linked_edges,
        neighbors=neighbors,
        # Carry along full per-source details for merge history awareness
        merged_sources=agg.merged_sources,
    )


def build_node_comparison_input(
    index: MergeIndex, key_a: str, key_b: str
) -> NodeComparisonInput:
    a = index.nodes[key_a]
    b = index.nodes[key_b]
    return NodeComparisonInput(node_a=_node_view(a, index), node_b=_node_view(b, index))


def build_node_set_comparison_input(
    index: MergeIndex, keys: list[str]
) -> NodeSetComparisonInput:
    views: list[NodeViewForComparison] = []
    for k in keys:
        if k in index.nodes:
            views.append(_node_view(index.nodes[k], index))
    return NodeSetComparisonInput(nodes=views)
