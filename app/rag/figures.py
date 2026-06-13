"""图表召回：对 data/figures.jsonl 的图注做 bge-m3 向量检索，按问题语义返回最相关的图。

图注嵌入缓存到 data/figures.npy（行数与 figures.jsonl 不符时自动重建）。
图本身按需由 /figure-image 端点用 bbox 从 PDF 裁剪渲染，无需预生成图片。
"""
import json
from functools import lru_cache

import numpy as np

from app.config import DATA_DIR
from app.rag.embedder import embed_query, embed_texts

FIGURES_PATH = DATA_DIR / "figures.jsonl"
EMB_CACHE = DATA_DIR / "figures.npy"


@lru_cache(maxsize=1)
def _load() -> tuple[list[dict], np.ndarray]:
    if not FIGURES_PATH.exists():
        return [], np.zeros((0, 0), dtype="float32")
    figs = [json.loads(l) for l in FIGURES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    if EMB_CACHE.exists():
        mat = np.load(EMB_CACHE)
        if mat.shape[0] == len(figs):
            return figs, mat
    mat = np.asarray(embed_texts([f["caption"] for f in figs]), dtype="float32")
    np.save(EMB_CACHE, mat)
    return figs, mat


def search_figures(query: str, top_k: int = 3, min_score: float = 0.55) -> list[dict]:
    """返回与问题最相关的图（图注余弦相似度≥min_score），含裁剪所需的 source/page/bbox。"""
    figs, mat = _load()
    if not figs:
        return []
    q = np.asarray(embed_query(query), dtype="float32")
    scores = mat @ q  # 归一化向量内积=余弦
    out = []
    for i in np.argsort(-scores)[:top_k]:
        if scores[i] < min_score:
            break
        out.append({**figs[i], "score": float(scores[i])})
    return out
