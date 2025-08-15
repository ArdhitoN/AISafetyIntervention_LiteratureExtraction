from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

from pydantic import BaseModel, ValidationError, Field
from .prompts import OutputSchema

from .merge_types import (
    NodeAggregate,
    LinkedEdgeSummary,
    SourceMetadata,
    NodeAttributesBySource,
    EdgeAggregate,
    EdgeAttributesBySource,
)

logger = logging.getLogger(__name__)


class MergeIndex(BaseModel):
    """Aggregated, key-addressable view over nodes collected from outputs."""

    nodes: Dict[str, NodeAggregate]
    edges: Dict[str, EdgeAggregate] = Field(default_factory=dict)


def _node_key(canonical_name: str) -> str:
    return canonical_name.strip().lower()


def _iter_valid_output_files(output_dir: Path) -> Iterable[Path]:
    return sorted(p for p in output_dir.glob("*.json") if "raw_response" not in p.name)


def _load_output(path: Path) -> OutputSchema | None:
    try:
        return OutputSchema.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError) as e:
        logger.debug(f"Skipping invalid JSON file {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error loading {path}: {e}")
        return None


def _summarize_edge(
    edge, source_node_key: str, target_node_key: str, source: SourceMetadata
) -> LinkedEdgeSummary:
    return LinkedEdgeSummary(
        edge_type=edge.type,
        rationale=edge.rationale,
        confidence=edge.confidence,
        source_node_key=source_node_key,
        target_node_key=target_node_key,
        source=source,
    )


def _upsert_edge_aggregate(
    aggregates: Dict[str, EdgeAggregate], edge, source: SourceMetadata, target_key: str
) -> EdgeAggregate:
    agg = _get_or_create_edge_aggregate(aggregates, edge)
    _append_edge_global_lists(agg, edge, source, target_key)
    _upsert_edge_attributes_by_source(agg, edge, source, target_key)
    return agg


def _get_or_create_edge_aggregate(
    aggregates: Dict[str, EdgeAggregate], edge
) -> EdgeAggregate:
    edge_key = edge.type
    if edge_key in aggregates:
        return aggregates[edge_key]
    aggregates[edge_key] = EdgeAggregate(
        edge_key=edge_key,
        edge_type=edge.type,
        text=edge.type,
        node_pairs=[],
        rationales=[],
        confidence_samples=[],
        sources=[],
        attributes_by_source=[],
    )
    return aggregates[edge_key]


def _append_edge_global_lists(
    agg: EdgeAggregate, edge, source: SourceMetadata, target_key: str
) -> None:
    agg.node_pairs.append((f"paper:{source.paper_id}", target_key))
    agg.rationales.append(edge.rationale)
    agg.confidence_samples.append(edge.confidence)
    agg.sources.append(source)


def _upsert_edge_attributes_by_source(
    agg: EdgeAggregate, edge, source: SourceMetadata, target_key: str
) -> None:
    sid = source.paper_id
    for entry in agg.attributes_by_source:
        if entry.paper_id == sid:
            _merge_edge_attr_entry(entry, edge, sid, target_key)
            return
    agg.attributes_by_source.append(
        EdgeAttributesBySource(
            paper_id=sid,
            doi=getattr(source, "doi", None),
            node_pairs=[(f"paper:{sid}", target_key)],
            rationales=[edge.rationale],
            confidence_samples=[edge.confidence],
        )
    )


def _merge_edge_attr_entry(
    entry: EdgeAttributesBySource, edge, sid: str, target_key: str
) -> None:
    entry.node_pairs.append((f"paper:{sid}", target_key))
    entry.rationales.append(edge.rationale)
    entry.confidence_samples.append(edge.confidence)


def _upsert_node_aggregate(
    aggregates: Dict[str, NodeAggregate], node, source: SourceMetadata
) -> NodeAggregate:
    try:
        key = _node_key(node.canonical_name or node.name)
        if key not in aggregates:
            aggregates[key] = _create_node_aggregate(node=node, key=key, source=source)
            return aggregates[key]

        agg = aggregates[key]
        _merge_node_aggregate_fields(agg, node, source)
        _upsert_node_attributes_by_source(agg, node, source)
        return agg
    except Exception as e:
        logger.debug(f"Error upserting node {getattr(node, 'name', 'unknown')}: {e}")
        # Return a minimal aggregate to continue processing
        key = _node_key(getattr(node, "name", "unknown"))
        if key not in aggregates:
            aggregates[key] = NodeAggregate(
                node_key=key,
                text=getattr(node, "name", "unknown"),
                canonical_text=getattr(node, "name", "unknown"),
                aliases=[],
                notes=[],
                confidence_samples=[],
                linked_edges=[],
                sources=[source],
            )
        return aggregates[key]


def _create_node_aggregate(node, key: str, source: SourceMetadata) -> NodeAggregate:
    sid = source.paper_id
    agg = NodeAggregate(
        node_key=key,
        text=node.name,
        canonical_text=node.canonical_name or node.name,
        aliases=list(node.aliases or []),
        notes=[node.notes] if getattr(node, "notes", None) else [],
        confidence_samples=[node.confidence] if node.confidence is not None else [],
        linked_edges=[],
        sources=[source],
    )
    _seed_node_attributes_by_source(agg, node, source, sid)
    return agg


def _merge_node_aggregate_fields(
    agg: NodeAggregate, node, source: SourceMetadata
) -> None:
    if (
        node.canonical_name
        and node.canonical_name.lower() == agg.canonical_text.lower()
    ):
        agg.text = node.name
    agg.aliases = _merge_alias_lists(agg.aliases, node.aliases or [])
    note_text = getattr(node, "notes", None)
    if note_text and note_text not in agg.notes:
        agg.notes.append(note_text)
    if node.confidence is not None:
        agg.confidence_samples.append(node.confidence)
    agg.sources.append(source)


def _merge_alias_lists(existing: List[str], new_aliases: List[str]) -> List[str]:
    alias_set = set(existing)
    alias_set.update(new_aliases)
    return list(alias_set)


def _upsert_node_attributes_by_source(
    agg: NodeAggregate, node, source: SourceMetadata
) -> None:
    sid = source.paper_id
    for entry in agg.attributes_by_source:
        if entry.paper_id == sid:
            _merge_node_attr_entry(entry, node, sid)
            return
    agg.attributes_by_source.append(
        NodeAttributesBySource(
            paper_id=sid,
            doi=getattr(source, "doi", None),
            text=node.name,
            canonical_text=node.canonical_name or node.name,
            aliases=list(node.aliases or []),
            notes=[node.notes] if getattr(node, "notes", None) else [],
            confidence=node.confidence,
        )
    )


def _merge_node_attr_entry(entry: NodeAttributesBySource, node, sid: str) -> None:
    entry.text = node.name or entry.text
    entry.canonical_text = node.canonical_name or entry.canonical_text
    if node.aliases:
        entry.aliases = _merge_alias_lists(entry.aliases, list(node.aliases))
    if getattr(node, "notes", None):
        if node.notes not in entry.notes:
            entry.notes.append(node.notes)
    if node.confidence is not None:
        entry.confidence = node.confidence


def _seed_node_attributes_by_source(
    agg: NodeAggregate, node, source: SourceMetadata, sid: str
) -> None:
    agg.attributes_by_source.append(
        NodeAttributesBySource(
            paper_id=sid,
            doi=getattr(source, "doi", None),
            text=node.name,
            canonical_text=node.canonical_name or node.name,
            aliases=list(node.aliases or []),
            notes=[node.notes] if getattr(node, "notes", None) else [],
            confidence=node.confidence,
        )
    )


def build_merge_index(output_dir: Path) -> MergeIndex:
    """Aggregate nodes from validated outputs into a compact index.

    Current schema has edges with only target_node (source is implicit paper).
    LinkedEdgeSummary preserves this relationship and is ready for future
    node-to-node edges when the extraction schema evolves.
    """
    node_aggregates: Dict[str, NodeAggregate] = {}
    edge_aggregates: Dict[str, EdgeAggregate] = {}

    for json_path in _iter_valid_output_files(output_dir):
        _process_output_file(json_path, node_aggregates, edge_aggregates)

    return MergeIndex(nodes=node_aggregates, edges=edge_aggregates)


def _process_output_file(
    json_path: Path,
    node_aggregates: Dict[str, NodeAggregate],
    edge_aggregates: Dict[str, EdgeAggregate],
) -> None:
    data = _load_output(json_path)
    if data is None:
        return

    source_metadata = _source_from_path(json_path)

    for edge in data.edges:
        _process_edge_record(
            edge, source_metadata, node_aggregates, edge_aggregates, json_path
        )

    _add_neighbors_for_paper(data, node_aggregates, json_path)


def _process_edge_record(
    edge,
    source_metadata: SourceMetadata,
    node_aggregates: Dict[str, NodeAggregate],
    edge_aggregates: Dict[str, EdgeAggregate],
    context_path: Path,
) -> None:
    try:
        target_node = edge.target_node
        target_key = _node_key(target_node.canonical_name or target_node.name)

        source_key = f"paper:{source_metadata.paper_id}"

        target_agg = _upsert_node_aggregate(
            node_aggregates, target_node, source_metadata
        )

        edge_summary = _summarize_edge(
            edge=edge,
            source_node_key=source_key,
            target_node_key=target_key,
            source=source_metadata,
        )
        target_agg.linked_edges.append(edge_summary)

        _upsert_edge_aggregate(edge_aggregates, edge, source_metadata, target_key)
    except Exception as e:
        logger.debug(f"Error processing edge in {context_path}: {e}")


def _add_neighbors_for_paper(
    data: OutputSchema,
    node_aggregates: Dict[str, NodeAggregate],
    context_path: Path,
) -> None:
    try:
        paper_node_keys = _collect_paper_node_keys(data)
        unique_keys = list(dict.fromkeys(paper_node_keys))
        for i, k in enumerate(unique_keys):
            neighbors = unique_keys[:i] + unique_keys[i + 1 :]
            agg = node_aggregates.get(k)
            if not agg:
                continue
            neighbor_set = set(agg.neighbor_node_keys)
            neighbor_set.update(neighbors)
            agg.neighbor_node_keys = list(neighbor_set)
    except Exception as e:
        logger.debug(f"Error building neighbor lists for {context_path}: {e}")


def _collect_paper_node_keys(data: OutputSchema) -> list[str]:
    keys: list[str] = []
    for edge in data.edges:
        try:
            keys.append(_target_key_from_node(edge.target_node))
        except Exception:
            continue
    return keys


def _source_from_path(json_path: Path) -> SourceMetadata:
    # TODO: enrich with DOI/title/section when available upstream
    return SourceMetadata(paper_id=json_path.stem)


def _target_key_from_node(node) -> str:
    return _node_key(node.canonical_name or node.name)
