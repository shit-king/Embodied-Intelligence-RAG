import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

metas = [json.loads(l) for l in open("data/faiss/meta.jsonl", encoding="utf-8") if l.strip()]
charts = [m for m in metas if m.get("kind") == "chart"]
print("索引中图表块:", len(charts), "/", len(metas))

from app.rag.reranker import rerank
from app.rag.retriever import hybrid_recall

q = "具身智能机器人SWOT分析的优势和劣势是什么"
hits = rerank(q, hybrid_recall(q, 12))[:5]
for h in hits:
    print(f"{h['kind']:>5} {h['score']:.3f}《{h['source'][:32]}》p{h['page']}")
