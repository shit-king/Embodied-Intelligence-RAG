from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import EMBEDDING_DEVICE

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL, device=EMBEDDING_DEVICE, max_length=512)


def rerank(question: str, hits: list[dict]) -> list[dict]:
    """对候选块按与问题的相关性精排，score替换为sigmoid后的rerank分（0-1）。"""
    if not hits:
        return hits
    # 单标签CrossEncoder默认sigmoid激活，输出0-1
    scores = get_reranker().predict(
        [(question, h["text"]) for h in hits], batch_size=8
    )
    for h, s in zip(hits, scores):
        h["score"] = float(s)
    return sorted(hits, key=lambda h: h["score"], reverse=True)
