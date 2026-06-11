import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import PROJECT_ROOT
from app.rag.agent import stream_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.rag.bm25 import get_bm25
    from app.rag.embedder import get_model
    from app.rag.reranker import get_reranker

    get_model()
    get_reranker()
    get_bm25()
    yield


app = FastAPI(title="具身智能RAG问答", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    def gen():
        for evt in stream_agent(req.question):
            if evt["type"] == "token":
                yield f"data: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
            else:
                yield f"event: {evt['type']}\ndata: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "web" / "index.html")
