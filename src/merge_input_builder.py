from __future__ import annotations

from .merge_types import (
    NodeComparisonInput,
    NodeViewForComparison,
    NodeAggregate,
    NeighborNodeSummary,
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
        neighbors.append(
            NeighborNodeSummary(node_key=n.node_key, text=n.text, aliases=n.aliases)
        )

    return NodeViewForComparison(
        text=agg.text,
        aliases=agg.aliases,
        context=contexts + agg.notes,
        source_metadata=agg.sources,
        linked_edges=agg.linked_edges,
        neighbors=neighbors,
    )


def build_node_comparison_input(
    index: MergeIndex, key_a: str, key_b: str
) -> NodeComparisonInput:
    a = index.nodes[key_a]
    b = index.nodes[key_b]
    return NodeComparisonInput(node_a=_node_view(a, index), node_b=_node_view(b, index))
