"""混合召回：向量+BM25双路检索，RRF（倒数排名融合）合并。"""
from app.rag.bm25 import get_bm25
from app.rag.embedder import embed_query
from app.rag.vectorstore import get_vector_store

RRF_K = 60
RECALL_PER_ROUTE = 15


def _key(hit: dict) -> tuple:
    return (hit["source"], hit["page"], hit["text"][:80])


def hybrid_recall(query: str, top_k: int) -> list[dict]:
    vec_hits = get_vector_store().search(embed_query(query), RECALL_PER_ROUTE)
    bm_hits = get_bm25().search(query, RECALL_PER_ROUTE)
    fused: dict[tuple, dict] = {}
    for hits in (vec_hits, bm_hits):
        for rank, hit in enumerate(hits):
            entry = fused.setdefault(_key(hit), {**hit, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank + 1)
    ranked = sorted(fused.values(), key=lambda h: h["score"], reverse=True)
    return ranked[:top_k]
