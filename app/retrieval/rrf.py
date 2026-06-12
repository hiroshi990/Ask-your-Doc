"""Reciprocal Rank Fusion for combining dense and sparse retrieval results."""

from typing import Any


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked lists using RRF.
    score(d) = sum(1 / (k + rank(d))) across all lists
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = {**item}
            else:
                existing = chunk_map[chunk_id]
                for score_key in ("dense_score", "sparse_score"):
                    if item.get(score_key) is not None:
                        existing[score_key] = item[score_key]

    fused = []
    for chunk_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = chunk_map[chunk_id]
        entry["rrf_score"] = rrf_score
        fused.append(entry)

    return fused


def deduplicate_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate chunks by chunk_id, keeping highest RRF score."""
    seen: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid not in seen or chunk.get("rrf_score", 0) > seen[cid].get("rrf_score", 0):
            seen[cid] = chunk
    return list(seen.values())
