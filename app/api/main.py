import hashlib
import json
from contextlib import asynccontextmanager

import fitz
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from app.config import DATA_DIR, PDF_DIR, PROJECT_ROOT
from app.rag.agent import stream_agent

PAGE_CACHE = DATA_DIR / "page_cache"


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
    mode: str = "fast"


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    def gen():
        for evt in stream_agent(req.question, req.mode):
            if evt["type"] == "token":
                yield f"data: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
            else:
                yield f"event: {evt['type']}\ndata: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "web" / "index.html")


@app.get("/page-image")
def page_image(source: str, page: int) -> Response:
    """渲染指定报告某页为图片，供来源卡片展示原页（磁盘缓存）。"""
    if any(c in source for c in ("/", "\\", "..")):
        raise HTTPException(400, "非法来源名")
    pdf_path = PDF_DIR / f"{source}.pdf"
    if not pdf_path.exists():
        raise HTTPException(404, "报告不存在")
    PAGE_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{source}|{page}".encode()).hexdigest()
    cached = PAGE_CACHE / f"{key}.jpg"
    if not cached.exists():
        with fitz.open(pdf_path) as doc:
            if not 1 <= page <= len(doc):
                raise HTTPException(404, "页码越界")
            pg = doc[page - 1]
            # 大开本页限制渲染尺寸，避免生成数MB的超大图
            zoom = min(2.0, 2200 / max(pg.rect.width, pg.rect.height))
            pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            cached.write_bytes(pix.tobytes("jpeg", jpg_quality=80))
    return Response(cached.read_bytes(), media_type="image/jpeg")
