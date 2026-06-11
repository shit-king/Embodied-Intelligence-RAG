"""检索质量对比：纯向量 vs 混合召回+rerank。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.rag.embedder import embed_query
from app.rag.reranker import rerank
from app.rag.retriever import hybrid_recall
from app.rag.vectorstore import get_vector_store

QUERIES = [
    "宇树Unitree Dex5灵巧手有什么特点",          # 专有名词，BM25强项
    "PEEK材料在具身智能中的应用",                # 缩写词
    "关节占人形机器人制造成本的比例是多少",       # 数字事实
]

for q in QUERIES:
    print(f"\n{'=' * 40}\n查询：{q}")
    print("--- 纯向量 top3 ---")
    for h in get_vector_store().search(embed_query(q), 3):
        print(f"  {h['score']:.3f}《{h['source'][:30]}》p{h['page']}")
    print("--- 混合召回+rerank top3 ---")
    for h in rerank(q, hybrid_recall(q, 12))[:3]:
        print(f"  {h['score']:.3f}《{h['source'][:30]}》p{h['page']}")
