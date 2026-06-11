"""Agentic RAG：LangGraph编排的路由→拆解→多路检索→自修正→引用合成流程。"""
import json
from functools import lru_cache
from typing import Iterator, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import TOP_K
from app.rag.embedder import embed_query
from app.rag.qa import SYSTEM_PROMPT, USER_PROMPT, build_context, get_llm
from app.rag.vectorstore import get_vector_store

MIN_SCORE = 0.5          # 最高分低于此值视为检索失败，触发改写重试
SUB_QUERY_TOP_K = 4      # 拆解后每个子问题的召回数
MAX_MERGED_HITS = 10     # 合并后送入生成的上下文块上限

ROUTE_PROMPT = """判断用户问题属于哪类，只输出JSON：{{"route": "simple"或"complex"}}

- simple：单一主题的事实查询，一次检索可回答。如"2025年市场规模是多少"
- complex：涉及对比、综合、多个主题或多步推理。如"对比A和B的预测差异"、"X的现状、瓶颈和趋势"

用户问题：{question}"""

DECOMPOSE_PROMPT = """将复杂问题拆解为2-4个适合向量检索的独立子查询，只输出JSON：
{{"sub_queries": ["子查询1", "子查询2"]}}

要求：每个子查询自包含（不用代词指代）、聚焦单一主题、保留关键实体名。

用户问题：{question}"""

REWRITE_PROMPT = """以下查询在行业报告知识库中检索效果差，请改写它们以提高召回：
换用行业报告中更常见的表述、补充同义词、拆开复合概念。只输出JSON：
{{"queries": ["改写后查询1", "改写后查询2"]}}

原查询：{queries}"""


class AgentState(TypedDict, total=False):
    question: str
    route: str
    sub_queries: list[str]
    hits: list[dict]
    retried: bool
    answer: str


def _llm_json(prompt: str) -> dict:
    llm = get_llm(streaming=False).bind(response_format={"type": "json_object"})
    try:
        return json.loads(llm.invoke(prompt).content)
    except (json.JSONDecodeError, KeyError):
        return {}


def route_node(state: AgentState) -> AgentState:
    result = _llm_json(ROUTE_PROMPT.format(question=state["question"]))
    return {"route": result.get("route", "simple")}


def decompose_node(state: AgentState) -> AgentState:
    result = _llm_json(DECOMPOSE_PROMPT.format(question=state["question"]))
    sub_queries = [q for q in result.get("sub_queries", []) if isinstance(q, str)][:4]
    return {"sub_queries": sub_queries}


def retrieve_node(state: AgentState) -> AgentState:
    queries = state.get("sub_queries") or [state["question"]]
    top_k = SUB_QUERY_TOP_K if len(queries) > 1 else TOP_K
    store = get_vector_store()
    merged, seen = [], set()
    for query in queries:
        for hit in store.search(embed_query(query), top_k):
            key = (hit["source"], hit["page"], hit["text"][:80])
            if key not in seen:
                seen.add(key)
                merged.append(hit)
    merged.sort(key=lambda h: h["score"], reverse=True)
    return {"hits": merged[:MAX_MERGED_HITS]}


def rewrite_node(state: AgentState) -> AgentState:
    queries = state.get("sub_queries") or [state["question"]]
    result = _llm_json(REWRITE_PROMPT.format(queries=json.dumps(queries, ensure_ascii=False)))
    rewritten = [q for q in result.get("queries", []) if isinstance(q, str)][:4]
    return {"sub_queries": rewritten or queries, "retried": True}


def synthesize_node(state: AgentState) -> AgentState:
    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(
        context=build_context(state["hits"]), question=state["question"]
    )
    answer = get_llm().invoke(prompt)
    return {"answer": answer.content}


def after_route(state: AgentState) -> str:
    return "decompose" if state["route"] == "complex" else "retrieve"


def after_retrieve(state: AgentState) -> str:
    hits = state.get("hits", [])
    weak = not hits or max(h["score"] for h in hits) < MIN_SCORE
    if weak and not state.get("retried"):
        return "rewrite"
    return "synthesize"


@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("decompose", decompose_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["decompose", "retrieve"])
    g.add_edge("decompose", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["rewrite", "synthesize"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("synthesize", END)
    return g.compile()


def stream_agent(question: str) -> Iterator[dict]:
    """产出事件流：{"type": "step"|"sources"|"token", "data": ...}"""
    graph = get_graph()
    for mode, chunk in graph.stream(
        {"question": question},
        stream_mode=["updates", "messages"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "synthesize" and msg.content:
                yield {"type": "token", "data": msg.content}
        elif mode == "updates":
            for node, delta in chunk.items():
                if node == "route":
                    label = "复杂问题，需拆解检索" if delta["route"] == "complex" else "简单问题，直接检索"
                    yield {"type": "step", "data": f"路由判断：{label}"}
                elif node == "decompose":
                    subs = "；".join(delta["sub_queries"])
                    yield {"type": "step", "data": f"拆解为{len(delta['sub_queries'])}个子查询：{subs}"}
                elif node == "rewrite":
                    yield {"type": "step", "data": "检索相关度不足，改写查询重试"}
                elif node == "retrieve":
                    yield {"type": "step", "data": f"召回{len(delta['hits'])}个相关片段"}
                    yield {"type": "sources", "data": delta["hits"]}
