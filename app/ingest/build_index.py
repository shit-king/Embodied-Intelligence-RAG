"""读取解析结果，切块、向量化并写入向量库，支持断点续跑。"""
import json
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, PARSED_DIR
from app.rag.embedder import embed_texts
from app.rag.vectorstore import get_vector_store

INDEXED_PATH = DATA_DIR / "indexed.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
)


def load_indexed() -> set:
    if INDEXED_PATH.exists():
        return set(json.loads(INDEXED_PATH.read_text(encoding="utf-8")))
    return set()


def save_indexed(indexed: set) -> None:
    INDEXED_PATH.write_text(json.dumps(sorted(indexed)), encoding="utf-8")


CHARTS_PATH = DATA_DIR / "parsed_charts.jsonl"


def index_charts(store, indexed: set) -> None:
    """图表页VL转写文本入库：单批编码，元数据带kind=chart。"""
    if not CHARTS_PATH.exists():
        return
    records = [
        json.loads(line)
        for line in CHARTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    todo = [r for r in records if f"chart:{r['digest']}:{r['page']}" not in indexed]
    if not todo:
        return
    print(f"图表页待入库 {len(todo)} 页")
    ids, texts, metadatas = [], [], []
    for r in todo:
        for i, chunk in enumerate(splitter.split_text(r["text"])):
            ids.append(f"chart_{r['digest']}_{r['page']}_{i}")
            texts.append(chunk)
            metadatas.append({"source": r["source"], "page": r["page"], "kind": "chart"})
    if texts:
        embeddings = embed_texts(texts)
        store.add(ids, embeddings, texts, metadatas)
    indexed.update(f"chart:{r['digest']}:{r['page']}" for r in todo)
    save_indexed(indexed)
    print(f"图表页入库完成：{len(texts)} 块")


def main() -> None:
    store = get_vector_store()
    indexed = load_indexed()
    doc_files = sorted(p for p in PARSED_DIR.glob("*.json") if p.stem != "manifest")
    todo = [p for p in doc_files if p.stem not in indexed]
    print(f"共 {len(doc_files)} 份文档，待入库 {len(todo)} 份，库内现有 {store.count()} 块")

    for n, doc_file in enumerate(todo, 1):
        doc = json.loads(doc_file.read_text(encoding="utf-8"))
        source = doc["source"]
        ids, texts, metadatas = [], [], []
        for page in doc["pages"]:
            for i, chunk in enumerate(splitter.split_text(page["text"])):
                ids.append(f"{doc_file.stem}_{page['page']}_{i}")
                texts.append(chunk)
                metadatas.append({"source": source, "page": page["page"]})
        if texts:
            print(f"[{n}/{len(todo)}] {source}: {len(texts)} 块")
            embeddings = embed_texts(texts)
            store.add(ids, embeddings, texts, metadatas)
        indexed.add(doc_file.stem)
        save_indexed(indexed)

    index_charts(store, indexed)
    print(f"\n完成，库内共 {store.count()} 块")


if __name__ == "__main__":
    main()
