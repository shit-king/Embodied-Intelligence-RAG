"""评估打分阶段（.venv-eval运行）：读取runs/*.jsonl，RAGAS用DeepSeek裁判打分，输出对比报告。

指标（均无需人工标注答案）：
- faithfulness：回答中的论断有多少能被检索上下文支持（忠实度，反幻觉）
- answer_relevancy：回答与问题的相关程度
- context_precision：检索到的上下文中相关内容的占比与排序质量
"""
import json
import os
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent

for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

METRIC_COLS = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]
METRIC_LABELS = {"faithfulness": "Faithfulness", "answer_relevancy": "AnswerRelevancy",
                 "llm_context_precision_without_reference": "ContextPrecision"}


def load_records(name: str) -> list[dict]:
    path = EVAL_DIR / "runs" / f"{name}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(records, key=lambda r: r["id"])


def main() -> None:
    judge = LangchainLLMWrapper(ChatOpenAI(
        model="deepseek-chat", base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"], temperature=0,
    ))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ))
    # strictness=1：DeepSeek不支持n>1采样，反推问题只生成一次
    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithoutReference()]

    frames = {}
    for name in ["baseline", "agent"]:
        records = load_records(name)
        ds = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r["contexts"],
                response=r["answer"],
            )
            for r in records
        ])
        print(f"=== 评估 {name}（{len(records)}条）===")
        run_config = RunConfig(timeout=600, max_retries=15, max_workers=4)
        df = evaluate(
            dataset=ds, metrics=metrics, llm=judge, embeddings=emb, run_config=run_config
        ).to_pandas()
        df["category"] = [r["category"] for r in records]
        df["id"] = [r["id"] for r in records]
        df.to_csv(EVAL_DIR / "runs" / f"scores_{name}.csv", index=False, encoding="utf-8-sig")
        frames[name] = df

    lines = ["# RAGAS 评估报告：v0.1基线 vs v0.3完整链路", "",
             "- 基线：纯向量top6 + 直接生成（v0.1）",
             "- 完整链路：路由/拆解 + 混合召回 + rerank精排 + 合成（v0.3）",
             f"- 评估集：{len(frames['baseline'])}个问题（事实/专有名词/数字/跨文档综合），裁判模型 deepseek-chat", "",
             "## 总体得分", "",
             "| 指标 | 基线 | 完整链路 | 提升 |", "|---|---|---|---|"]
    for col in METRIC_COLS:
        b, a = frames["baseline"][col].mean(), frames["agent"][col].mean()
        lines.append(f"| {METRIC_LABELS[col]} | {b:.3f} | {a:.3f} | {a - b:+.3f} |")

    lines += ["", "## 分类别得分", ""]
    for col in METRIC_COLS:
        lines += [f"**{METRIC_LABELS[col]}**", "", "| 类别 | 基线 | 完整链路 | 提升 |", "|---|---|---|---|"]
        for cat in frames["baseline"]["category"].unique():
            b = frames["baseline"][frames["baseline"]["category"] == cat][col].mean()
            a = frames["agent"][frames["agent"]["category"] == cat][col].mean()
            lines.append(f"| {cat} | {b:.3f} | {a:.3f} | {a - b:+.3f} |")
        lines.append("")

    (EVAL_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
