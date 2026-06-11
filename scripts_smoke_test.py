import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.rag.qa import answer, retrieve

q = "2025年具身智能市场规模预测是多少？"
print("=== 检索结果 ===")
for h in retrieve(q):
    print(f"{h['score']:.3f} 《{h['source'][:36]}》 p{h['page']}")

print("\n=== 问答 ===")
hits, ans = answer(q)
print(ans)
