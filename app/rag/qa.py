import os
from typing import Iterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, TOP_K
from app.rag.embedder import embed_query
from app.rag.vectorstore import get_vector_store

SYSTEM_PROMPT = """你是具身智能行业研究助手，基于提供的研究报告片段回答用户问题。

要求：
1. 只依据提供的资料回答，资料中没有的信息明确说"资料中未提及"，不要编造。
2. 引用资料时在句末标注来源编号，格式如 [来源1]、[来源2][来源3]。
3. 涉及数据（市场规模、增速、融资额等）必须标注来源。
4. 多份报告观点不一致时，分别列出并指明各自来源。
5. 用中文回答，结构清晰。"""

USER_PROMPT = """资料：
{context}

问题：{question}"""


def get_llm(streaming: bool = True) -> ChatOpenAI:
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=0.3,
        streaming=streaming,
    )


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    return get_vector_store().search(embed_query(question), top_k)


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        tag = "（图表页转写）" if h.get("kind") == "chart" else ""
        blocks.append(f"【来源{i}】《{h['source']}》第{h['page']}页{tag}：\n{h['text']}")
    return "\n\n".join(blocks)


def answer_stream(question: str) -> tuple[list[dict], Iterator[str]]:
    """返回 (检索结果, 回答token流)。"""
    hits = retrieve(question)
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
    )
    chain = prompt | get_llm()

    def stream() -> Iterator[str]:
        for chunk in chain.stream(
            {"context": build_context(hits), "question": question}
        ):
            if chunk.content:
                yield chunk.content

    return hits, stream()


def answer(question: str) -> tuple[list[dict], str]:
    hits, stream = answer_stream(question)
    return hits, "".join(stream)
