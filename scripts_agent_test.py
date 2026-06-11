import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.rag.agent import stream_agent

for q in [
    "智元机器人是做什么的？",
    "对比不同报告对人形机器人量产时间表的预测，并分析数据瓶颈对量产的影响",
]:
    print(f"\n{'=' * 30}\n问题：{q}\n{'=' * 30}")
    tokens = []
    for evt in stream_agent(q):
        if evt["type"] == "step":
            print(f"[STEP] {evt['data']}")
        elif evt["type"] == "sources":
            for i, h in enumerate(evt["data"], 1):
                print(f"  来源{i} {h['score']:.3f}《{h['source'][:28]}》p{h['page']}")
        else:
            tokens.append(evt["data"])
    answer = "".join(tokens)
    print(f"\n回答（前600字）：\n{answer[:600]}")
