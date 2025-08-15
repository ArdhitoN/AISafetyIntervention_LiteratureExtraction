from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    """Minimal source metadata available from extractions.

    Fields are optional because many are not available from the current pipeline.
    """

    paper_id: str = Field(
        ..., description="Identifier derived from the output filename (stem)"
    )
    doi: Optional[str] = Field(
        default=None, description="Digital Object Identifier, if known"
    )
    title: Optional[str] = Field(default=None, description="Paper title if known")
    section: Optional[str] = Field(default=None, description="Section heading if known")
    paragraph_id: Optional[str] = Field(
        default=None, description="Paragraph identifier if known"
    )


class LinkedEdgeSummary(BaseModel):
    """Summarized view of an edge touching a node for LLM context with directionality."""

    edge_type: str
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_node_key: str = Field(..., description="Key of the source node")
    target_node_key: str = Field(..., description="Key of the target node")
    source: SourceMetadata

    def get_context_for_node(self, node_key: str) -> str:
        """Generate contextual description based on node's role in the edge."""
        if node_key == self.source_node_key:
            return f"[{self.edge_type} -> {self.target_node_key}] {self.rationale}"
        elif node_key == self.target_node_key:
            return f"[{self.source_node_key} -> {self.edge_type}] {self.rationale}"
        else:
            return f"[{self.edge_type}] {self.rationale}"  # fallback


class NodeAggregate(BaseModel):
    """Aggregate information for a node across multiple papers/edges."""

    node_key: str = Field(
        ...,
        description="Stable key used to identify this aggregate (e.g., canonical_name lowercased)",
    )
    text: str = Field(..., description="Primary surface name for the node (name)")
    canonical_text: str = Field(
        ..., description="Canonical name provided by extraction"
    )
    aliases: List[str] = Field(default_factory=list)
    notes: List[str] = Field(
        default_factory=list, description="Collected node notes across occurrences"
    )
    confidence_samples: List[float] = Field(default_factory=list)
    linked_edges: List[LinkedEdgeSummary] = Field(default_factory=list)
    sources: List[SourceMetadata] = Field(default_factory=list)
    # Nearest neighbor node keys (A.IN1+) aggregated across sources
    neighbor_node_keys: List[str] = Field(
        default_factory=list,
        description="Unique list of immediate neighbor node keys gathered from the graph context",
    )

    # Array form for provenance-aware recursive merging (one entry per source)
    attributes_by_source: List["NodeAttributesBySource"] = Field(default_factory=list)


class EdgeAggregate(BaseModel):
    """Aggregate information for an edge across multiple papers."""

    edge_key: str
    edge_type: str
    text: str = Field(..., description="Edge-level description if available")
    node_pairs: List[tuple[str, str]] = Field(
        default_factory=list,
        description="List of (source_node_key, target_node_key) tuples preserving exact pairings",
    )
    rationales: List[str] = Field(default_factory=list)
    confidence_samples: List[float] = Field(default_factory=list)
    sources: List[SourceMetadata] = Field(default_factory=list)
    # Array form for per-source provenance
    attributes_by_source: List["EdgeAttributesBySource"] = Field(default_factory=list)


class NodeViewForComparison(BaseModel):
    text: str
    aliases: List[str]
    context: List[str] = Field(
        default_factory=list,
        description="Context strings: node notes and related rationales",
    )
    source_metadata: List[SourceMetadata]
    linked_edges: List[LinkedEdgeSummary]
    # Immediate neighbor nodes included for robustness
    neighbors: List["NeighborNodeSummary"] = Field(default_factory=list)


class EdgeViewForComparison(BaseModel):
    text: str
    node_pairs: List[tuple[str, str]] = Field(
        description="List of (source_node_key, target_node_key) tuples for this edge type"
    )
    context: List[str] = Field(default_factory=list)
    source_metadata: List[SourceMetadata]


class NeighborNodeSummary(BaseModel):
    node_key: str
    text: str
    aliases: List[str] = Field(default_factory=list)


class NodeComparisonInput(BaseModel):
    node_a: NodeViewForComparison
    node_b: NodeViewForComparison


class NodeAttributesBySource(BaseModel):
    paper_id: str
    doi: Optional[str] = None
    text: Optional[str] = None
    canonical_text: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class EdgeAttributesBySource(BaseModel):
    paper_id: str
    doi: Optional[str] = None
    node_pairs: List[tuple[str, str]] = Field(default_factory=list)
    rationales: List[str] = Field(default_factory=list)
    confidence_samples: List[float] = Field(default_factory=list)
