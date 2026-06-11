"""BM25关键词检索：jieba分词，基于FAISS同源的meta.jsonl语料，分词结果缓存。"""
import pickle
from functools import lru_cache

import jieba
from rank_bm25 import BM25Okapi

from app.config import DATA_DIR
from app.rag.vectorstore import get_vector_store

BM25_CACHE = DATA_DIR / "bm25.pkl"


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.cut_for_search(text) if t.strip()]


class BM25Index:
    def __init__(self) -> None:
        self.metas = get_vector_store().metas
        if BM25_CACHE.exists():
            with open(BM25_CACHE, "rb") as f:
                cached = pickle.load(f)
        else:
            cached = None
        if cached is None or len(cached) != len(self.metas):
            corpus = [_tokenize(m["text"]) for m in self.metas]
            with open(BM25_CACHE, "wb") as f:
                pickle.dump(corpus, f)
        else:
            corpus = cached
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[dict]:
        scores = self.bm25.get_scores(_tokenize(query))
        order = scores.argsort()[::-1][:top_k]
        hits = []
        for idx in order:
            if scores[idx] <= 0:
                break
            m = self.metas[idx]
            hits.append(
                {
                    "text": m["text"],
                    "source": m["source"],
                    "page": m["page"],
                    "score": float(scores[idx]),
                }
            )
        return hits


@lru_cache(maxsize=1)
def get_bm25() -> BM25Index:
    return BM25Index()
