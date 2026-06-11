import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.config import PROJECT_ROOT
from app.rag import qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.rag.embedder import get_model

    get_model()
    yield


app = FastAPI(title="具身智能RAG问答", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    def gen():
        hits, stream = qa.answer_stream(req.question)
        yield f"event: sources\ndata: {json.dumps(hits, ensure_ascii=False)}\n\n"
        for token in stream:
            yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "web" / "index.html")
