### LLM-Assisted Graph Merging — Step 1: Prepare Inputs for the LLM

This document explains how the Step 1 code prepares compact, data-only inputs
for LLM comparisons between nodes and edges.
It focuses solely on preparing the data; there are no prompts, LLM calls, or merge actions here.


## Goals (Step 1)
- Provide the LLM enough context to decide if two or more nodes should be merged.
- Data included per comparison:
  - Node merged_text (current short title; raw node text if no merge yet)
  - Aliases / alternate names
  - Context / reasoning (why the node exists): node notes + rationales from linked edges
  - Provenance / merge history as `merged_sources` (per-source records with paper_id, optional doi/title/section/paragraph_id, original text, and lightweight context)
  - Linked edges (immediate connections with type, rationale, confidence, and source)
  - Immediate neighbor nodes (A.IN1+ and B.IN1+) as summaries including name, aliases, and lightweight context (neighbor notes + linked-edge rationales)


## Files Overview
- `src/merge_types.py`
  - `SourceMetadata`: identifies where a node/edge came from (currently `paper_id` from the output filename; placeholders for `title`, `section`, `paragraph_id`).
  - `LinkedEdgeSummary`: short record of an immediate connection (edge type, rationale, confidence, and source).
  - `NodeAggregate`: collected per-node data from all occurrences across outputs. Adds `neighbor_node_keys`, `merged_text`, and `merged_sources` (array entries per paper/raw node) to enable provenance-aware, recursive merges.
  - `NodeViewForComparison`: the per-node view handed to the LLM later (`merged_text`, aliases, context, linked edges, `neighbors` including context, and full `merged_sources` for merge history).
  - `NodeComparisonInput`: a pair of `NodeViewForComparison` (data-only) for A vs B.
  - `NodeSetComparisonInput`: an N-way comparison payload carrying a list of `NodeViewForComparison`.
  - Edge types include `EdgeViewForComparison` (which now includes `merged_sources`) for a later step.

- `src/merge_indexer.py`
  - `build_merge_index(output_dir)`: reads parsed `OutputSchema` JSON files from `output/`, aggregates nodes by canonical name (lowercased), merges aliases and notes, and collects linked edge rationales as context.
  - Produces `MergeIndex` with `nodes: Dict[node_key, NodeAggregate]`.
  - Populates `neighbor_node_keys` by co-occurrence within the same paper (all targets in a paper are considered 1-hop neighbors of each other). Only neighbor node keys are recorded; no neighbor edges are pulled.
  - Fills `merged_sources` with one entry per paper holding node attributes (text, canonical_text, aliases, notes, confidence, optional `title/section/paragraph_id`, and lightweight `context`).
  - Aggregates edges into `edges: Dict[edge_type, EdgeAggregate]`, also with `merged_sources` per paper (node_pairs, rationales, confidence_samples, optional `title/section/paragraph_id`) for recursive merging and provenance-aware edge analysis.

- `src/merge_input_builder.py`
  - `build_node_comparison_input(index, key_a, key_b)`: converts two aggregates to a data-only `NodeComparisonInput` (pairwise mode).
  - `build_node_set_comparison_input(index, keys)`: converts N aggregates to a data-only `NodeSetComparisonInput` (N-way mode).
  - The node view includes:
    - `merged_text`, `aliases`
    - `context` (node notes + "[EDGE_TYPE] rationale" for immediate connections)
    - `linked_edges`
    - `neighbors` (A.IN1+/B.IN1+ summaries, including neighbor context)
    - `merged_sources` (per-paper/raw-node attributes and context)

- `examples/walkthrough_prepare_llm_input.py`
  - Runnable script demonstrating the end-to-end Step 1 flow. Supports pairwise and N-way modes.


## How Requirements Are Met
- Node text: `NodeViewForComparison.merged_text`
- Aliases: `NodeViewForComparison.aliases`
- Context / reasoning: `NodeViewForComparison.context` combines node notes with connected edge rationales.
- Provenance / merge history: `NodeViewForComparison.merged_sources` carries per-paper/raw-node attributes and context.
- Linked edges: `NodeViewForComparison.linked_edges` contains `LinkedEdgeSummary` for each immediate connection.

There is deliberately no prompt or instruction text in Step 1. This is data-only.


## Walkthrough
- List available node keys and build a comparison payload:
```bash
uv run python examples/walkthrough_prepare_llm_input.py --list --limit 10
uv run python examples/walkthrough_prepare_llm_input.py
uv run python examples/walkthrough_prepare_llm_input.py --key-a <node_key_a> --key-b <node_key_b>
uv run python examples/walkthrough_prepare_llm_input.py --save output/node_comparison_example.json
# N-way comparison (provide any number of keys)
uv run python examples/walkthrough_prepare_llm_input.py --keys <key1> <key2> <key3> <key4>
```

The printed (or saved) JSON is the exact payload you can provide to an LLM in Step 2.


## How This Enables Step 2
- Candidate selection can embed `NodeAggregate` fields (canonical name, merged_text, aliases, top-k notes) and return top-N similar `node_key` pairs or sets.
- Merge execution uses stable `node_key`s to unify aliases/notes and repoint edges.
