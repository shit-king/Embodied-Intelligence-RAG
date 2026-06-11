"""向量库抽象层：MVP用FAISS实现，后续可平滑替换为Milvus。"""
import json
from abc import ABC, abstractmethod

import faiss
import numpy as np

from app.config import DATA_DIR, EMBEDDING_DIM

FAISS_DIR = DATA_DIR / "faiss"


class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None: ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """返回 [{text, source, page, score}]，score越大越相关。"""

    @abstractmethod
    def count(self) -> int: ...


class FaissVectorStore(VectorStore):
    def __init__(self) -> None:
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        self.index_path = FAISS_DIR / "index.bin"
        self.meta_path = FAISS_DIR / "meta.jsonl"
        if self.index_path.exists():
            # 经字节序列化读写，绕开faiss C++层不支持中文路径的问题
            self.index = faiss.deserialize_index(
                np.frombuffer(self.index_path.read_bytes(), dtype=np.uint8)
            )
            # 不用splitlines()：块文本含U+2028等分隔符会被错误拆行
            with open(self.meta_path, encoding="utf-8") as f:
                self.metas = [json.loads(line) for line in f if line.strip()]
        else:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.metas = []

    def add(self, ids, embeddings, texts, metadatas) -> None:
        self.index.add(np.asarray(embeddings, dtype=np.float32))
        with open(self.meta_path, "a", encoding="utf-8") as f:
            for id_, text, meta in zip(ids, texts, metadatas):
                record = {"id": id_, "text": text, **meta}
                self.metas.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.index_path.write_bytes(faiss.serialize_index(self.index).tobytes())

    def search(self, query_embedding, top_k) -> list[dict]:
        query = np.asarray([query_embedding], dtype=np.float32)
        scores, indices = self.index.search(query, top_k)
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            m = self.metas[idx]
            hits.append(
                {
                    "text": m["text"],
                    "source": m["source"],
                    "page": m["page"],
                    "kind": m.get("kind", "text"),
                    "score": float(score),
                }
            )
        return hits

    def count(self) -> int:
        return self.index.ntotal


def get_vector_store() -> VectorStore:
    return FaissVectorStore()
