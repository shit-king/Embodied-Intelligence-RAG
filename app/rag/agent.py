"""Agentic RAG：LangGraph编排的（历史改写→）路由→拆解→多路检索→自修正→引用合成流程。

多轮对话：MemorySaver checkpointer 按 thread_id 保存对话历史；condense 节点在检索前
把含指代/省略的追问改写成不依赖上下文的独立问题，再走原有检索链路。
"""
import json
from functools import lru_cache
from typing import Annotated, Iterator, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.rag.qa import SYSTEM_PROMPT, USER_PROMPT, build_context, get_llm
from app.rag.reranker import rerank
from app.rag.retriever import hybrid_recall

MIN_SCORE = 0.2          # rerank最高分低于此值视为检索失败，触发改写重试
RECALL_TOP_K = 12        # 每个查询混合召回的候选数
MAX_CANDIDATES = 30      # 送入reranker的候选上限
MAX_MERGED_HITS = 10     # 精排后送入生成的上下文块上限
HISTORY_TURNS = 3        # condense 改写时回看的最近对话轮数

CONDENSE_PROMPT = """根据对话历史，把用户的最新问题改写成一个不依赖上下文、可独立检索的完整问题：
- 把"它/这个/他们/该公司/上面提到的"等指代，替换成历史中对应的具体实体或主题
- 补全省略的主语或对象
- 若最新问题本身已完整、或与历史无关，则原样输出，不要画蛇添足
只输出改写后的问题本身，不要任何解释或前缀。

对话历史：
{history}

最新问题：{question}

改写后的独立问题："""

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
    question: str                 # 本轮用户原始问题
    standalone_question: str      # 历史感知改写后的独立问题（检索/合成均用它）
    mode: str                     # fast | thinking，仅影响最终合成；内部决策固定走fast
    route: str
    sub_queries: list[str]
    hits: list[dict]
    retried: bool
    answer: str
    history: Annotated[list, add_messages]  # 跨轮对话历史（checkpointer 持久化）


def _query(state: AgentState) -> str:
    return state.get("standalone_question") or state["question"]


def _llm_json(prompt: str) -> dict:
    llm = get_llm(streaming=False, mode="fast").bind(response_format={"type": "json_object"})
    try:
        return json.loads(llm.invoke(prompt).content)
    except (json.JSONDecodeError, KeyError):
        return {}


def _render_history(history: list) -> str:
    lines = []
    for m in history[-HISTORY_TURNS * 2:]:
        role = "用户" if isinstance(m, HumanMessage) else "助手"
        text = m.content if len(m.content) <= 200 else m.content[:200] + "…"
        lines.append(f"{role}：{text}")
    return "\n".join(lines)


def condense_node(state: AgentState) -> AgentState:
    """历史感知改写 + 重置上一轮残留的瞬态字段（防跨轮污染）。"""
    question = state["question"]
    history = state.get("history", [])
    standalone = question
    if history:
        prompt = CONDENSE_PROMPT.format(history=_render_history(history), question=question)
        try:
            standalone = get_llm(streaming=False, mode="fast").invoke(prompt).content.strip() or question
        except Exception:
            standalone = question
    return {"standalone_question": standalone, "retried": False, "sub_queries": []}


def route_node(state: AgentState) -> AgentState:
    result = _llm_json(ROUTE_PROMPT.format(question=_query(state)))
    return {"route": result.get("route", "simple")}


def decompose_node(state: AgentState) -> AgentState:
    result = _llm_json(DECOMPOSE_PROMPT.format(question=_query(state)))
    sub_queries = [q for q in result.get("sub_queries", []) if isinstance(q, str)][:4]
    return {"sub_queries": sub_queries}


def retrieve_node(state: AgentState) -> AgentState:
    queries = state.get("sub_queries") or [_query(state)]
    candidates, seen = [], set()
    for query in queries:
        for hit in hybrid_recall(query, RECALL_TOP_K):
            key = (hit["source"], hit["page"], hit["text"][:80])
            if key not in seen:
                seen.add(key)
                candidates.append(hit)
    # RRF分仅用于路内排序，跨查询合并后统一交给reranker按（改写后的）原问题精排
    ranked = rerank(_query(state), candidates[:MAX_CANDIDATES])
    return {"hits": ranked[:MAX_MERGED_HITS]}


def rewrite_node(state: AgentState) -> AgentState:
    queries = state.get("sub_queries") or [_query(state)]
    result = _llm_json(REWRITE_PROMPT.format(queries=json.dumps(queries, ensure_ascii=False)))
    rewritten = [q for q in result.get("queries", []) if isinstance(q, str)][:4]
    return {"sub_queries": rewritten or queries, "retried": True}


def synthesize_node(state: AgentState) -> AgentState:
    prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT.format(
        context=build_context(state["hits"]), question=_query(state)
    )
    answer = get_llm(mode=state.get("mode", "fast")).invoke(prompt)
    # 历史里存用户实际问的原话 + 本轮回答，供下一轮 condense 解析指代
    return {"answer": answer.content,
            "history": [HumanMessage(state["question"]), AIMessage(answer.content)]}


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
    g.add_node("condense", condense_node)
    g.add_node("route", route_node)
    g.add_node("decompose", decompose_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("synthesize", synthesize_node)
    g.add_edge(START, "condense")
    g.add_edge("condense", "route")
    g.add_conditional_edges("route", after_route, ["decompose", "retrieve"])
    g.add_edge("decompose", "retrieve")
    g.add_conditional_edges("retrieve", after_retrieve, ["rewrite", "synthesize"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("synthesize", END)
    return g.compile(checkpointer=MemorySaver())


def stream_agent(question: str, mode: str = "fast", thread_id: str = "default") -> Iterator[dict]:
    """产出事件流：{"type": "step"|"sources"|"figures"|"token", "data": ...}

    thread_id 标识一次对话会话，checkpointer 据此跨轮保留历史。
    """
    from app.rag.figures import search_figures

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    for stream_mode, chunk in graph.stream(
        {"question": question, "mode": mode},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if stream_mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "synthesize" and msg.content:
                yield {"type": "token", "data": msg.content}
        elif stream_mode == "updates":
            for node, delta in chunk.items():
                if node == "condense":
                    sq = delta.get("standalone_question", question)
                    if sq != question:
                        yield {"type": "step", "data": f"结合上文，理解为：{sq}"}
                    # 图表按（改写后的）独立问题召回，追问也能命中相关图
                    figures = search_figures(sq)
                    if figures:
                        yield {"type": "step", "data": f"按图注召回 {len(figures)} 张相关图表"}
                        yield {"type": "figures", "data": figures}
                elif node == "route":
                    label = "复杂问题，需拆解检索" if delta["route"] == "complex" else "简单问题，直接检索"
                    yield {"type": "step", "data": f"路由判断：{label}"}
                elif node == "decompose":
                    subs = "；".join(delta["sub_queries"])
                    yield {"type": "step", "data": f"拆解为{len(delta['sub_queries'])}个子查询：{subs}"}
                elif node == "rewrite":
                    yield {"type": "step", "data": "检索相关度不足，改写查询重试"}
                elif node == "retrieve":
                    yield {"type": "step", "data": f"向量+BM25双路召回，精排后取{len(delta['hits'])}个片段"}
                    yield {"type": "sources", "data": delta["hits"]}
