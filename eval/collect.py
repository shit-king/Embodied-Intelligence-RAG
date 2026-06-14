"""评估采集阶段（主venv运行）：对每个问题分别跑基线和完整链路，落盘JSONL。

- baseline：v0.1链路（纯向量top6 + 直接生成）
- agent：v0.3链路（路由/拆解 + 混合召回 + rerank + 合成）
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.rag import qa
from app.rag.agent import stream_agent

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"


def run_baseline(question: str) -> dict:
    hits, answer = qa.answer(question)
    return {"contexts": [h["text"] for h in hits], "answer": answer}


def run_agent(question: str) -> dict:
    contexts, tokens = [], []
    # 每题独立 thread_id：评估是单轮，避免 checkpointer 跨题历史串台污染 condense
    for evt in stream_agent(question, thread_id=f"eval-{uuid.uuid4()}"):
        if evt["type"] == "sources":
            contexts = [h["text"] for h in evt["data"]]
        elif evt["type"] == "token":
            tokens.append(evt["data"])
    return {"contexts": contexts, "answer": "".join(tokens)}


def main() -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    samples = [
        json.loads(line)
        for line in (EVAL_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for name, runner in [("baseline", run_baseline), ("agent", run_agent)]:
        out_path = RUNS_DIR / f"{name}.jsonl"
        done_ids = set()
        if out_path.exists():
            done_ids = {
                json.loads(line)["id"]
                for line in out_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        with open(out_path, "a", encoding="utf-8") as f:
            for s in samples:
                if s["id"] in done_ids:
                    continue
                print(f"[{name}] #{s['id']} {s['question'][:30]}")
                result = runner(s["question"])
                record = {**s, **result}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    print("采集完成")


if __name__ == "__main__":
    main()
