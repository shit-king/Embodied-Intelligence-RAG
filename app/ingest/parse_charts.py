"""图表/扫描页解析：渲染为图片 → VL模型转写为可检索文本，逐页落盘断点续跑。

用法：python app/ingest/parse_charts.py [--limit N]
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import fitz
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.config import DATA_DIR, PARSED_DIR, PDF_DIR, VL_BASE_URL, VL_MODEL

CHARTS_PATH = DATA_DIR / "parsed_charts.jsonl"
RENDER_ZOOM = 2.0
MAX_RETRIES = 3

PROMPT = """这是行业研究报告《{source}》第{page}页的截图，该页以图表/图片为主。\
请将页面内容完整转写为可检索的文字：
1. 页面标题与核心结论
2. 每个图表：标题、类型、坐标轴与图例含义，并尽量精确地列出全部数据点的数值（年份、数值、单位）
3. 表格逐行转为文字
4. 产业链图谱/架构图：列出各环节名称和代表公司
只转写页面上实际存在的内容，没有的板块直接跳过（不要写"无XX展示"之类的占位说明）。\
忽略页眉页脚和水印。直接输出转写内容，不要任何客套或说明。"""


def render_page_jpeg(pdf_path: Path, page_no: int) -> bytes:
    with fitz.open(pdf_path) as doc:
        pix = doc[page_no - 1].get_pixmap(matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM))
        return pix.tobytes("jpeg", jpg_quality=85)


def transcribe(client: OpenAI, image: bytes, source: str, page: int) -> str:
    b64 = base64.b64encode(image).decode()
    resp = client.chat.completions.create(
        model=VL_MODEL,
        temperature=0.1,
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT.format(source=source, page=page)},
            ],
        }],
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="本次最多解析的页数（0=不限）")
    args = parser.parse_args()

    manifest = json.loads((PARSED_DIR / "manifest.json").read_text(encoding="utf-8"))
    done = set()
    if CHARTS_PATH.exists():
        for line in CHARTS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["digest"], r["page"]))

    worklist = []
    for digest, info in manifest.items():
        for page in info.get("empty_pages", []):
            if (digest, page) not in done:
                pdf_path = PDF_DIR / f"{info['source']}.pdf"
                if pdf_path.exists():
                    worklist.append((digest, info["source"], pdf_path, page))

    total_pages = sum(len(m.get("empty_pages", [])) for m in manifest.values())
    print(f"图表页共 {total_pages}，已完成 {len(done)}，本次待处理 {len(worklist)}"
          + (f"（限{args.limit}页）" if args.limit else ""))
    if args.limit:
        worklist = worklist[: args.limit]

    # 显式超时：VL 接口偶发挂起，无超时会堵塞整个管道；超时后由下方重试循环兜底
    client = OpenAI(base_url=VL_BASE_URL, api_key=os.environ["VL_API_KEY"], timeout=90.0, max_retries=2)
    failed = 0
    with open(CHARTS_PATH, "a", encoding="utf-8") as f:
        for n, (digest, source, pdf_path, page) in enumerate(worklist, 1):
            print(f"[{n}/{len(worklist)}] 《{source[:30]}》p{page}")
            try:
                image = render_page_jpeg(pdf_path, page)
            except Exception as e:
                print(f"  渲染失败: {e}")
                failed += 1
                continue
            for attempt in range(MAX_RETRIES):
                try:
                    text = transcribe(client, image, source, page)
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        print(f"  失败（已重试{MAX_RETRIES}次）: {e}")
                        text = None
                    else:
                        time.sleep(5 * (attempt + 1))
            if text is None:
                failed += 1
                continue
            f.write(json.dumps(
                {"digest": digest, "source": source, "page": page, "text": text},
                ensure_ascii=False,
            ) + "\n")
            f.flush()

    print(f"\n完成，本次成功 {len(worklist) - failed}，失败 {failed}")


if __name__ == "__main__":
    main()
