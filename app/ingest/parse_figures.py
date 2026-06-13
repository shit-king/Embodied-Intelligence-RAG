"""图注+图区抽取：扫描每页文字层的「图N: 标题」图注，定位其相邻的图形区域，
落盘为 data/figures.jsonl（source/page/caption/bbox）。供"按问题召回图表并裁剪插入回答"使用。

裁剪策略（连续带 + 跳过）：图注通常紧贴图（上方或下方）。以图注为锚，向图形内容多的
一侧扩展到最近的正文/页眉脚/另一图注边界，并纳入紧邻的"资料来源"行。两侧都几乎无图形
内容时跳过（多为正文里的"图N"引用或纯表格），宁可漏掉也不产出含大段正文的脏图。

用法：python app/ingest/parse_figures.py
"""
import json
import re
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.config import DATA_DIR, PARSED_DIR, PDF_DIR

FIGURES_PATH = DATA_DIR / "figures.jsonl"
CAP_RE = re.compile(r"^(图表|图|Figure|Fig)\s*\.?\s*\d+")
SRC_RE = re.compile(r"^(资料来源|数据来源|资料及图表来源|来源|Source)")


def _lines(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type", 0) != 0:
            continue
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if t:
                out.append((fitz.Rect(ln["bbox"]), t))
    out.sort(key=lambda x: x[0].y0)
    return out


def detect_figures(page: fitz.Page) -> list[dict]:
    W, H = page.rect.width, page.rect.height
    mid = W / 2
    lines = _lines(page)
    caps = [(r, t) for r, t in lines if CAP_RE.match(t)]
    # 双栏判定：左右半区都有图注（研报常见的图网格页）
    two_col = any(r.x0 < mid for r, _ in caps) and any(r.x0 >= mid for r, _ in caps)

    def col_of(r: fitz.Rect) -> tuple[float, float]:
        if not two_col:
            return W * 0.05, W * 0.95
        return (W * 0.04, mid - 4) if r.x0 < mid else (mid + 4, W * 0.96)

    def kind(r: fitz.Rect, t: str) -> str:
        if CAP_RE.match(t):
            return "cap"
        if SRC_RE.match(t):
            return "src"
        if r.y0 < H * 0.06 or r.y1 > H * 0.95:
            return "edge"
        if r.width > (mid if two_col else W) * 0.55 and len(t) > 20:
            return "body"
        return "minor"  # 短文本：多为图内坐标轴/图例标签

    tagged = [(r, t, kind(r, t)) for r, t in lines]
    # 边界行：图注/正文/页眉脚 + "资料来源"行（来源行收尾一张图，必须作边界，否则会越过它误并相邻图）
    bounds = [r for r, t, k in tagged if k in ("body", "edge", "cap", "src")]

    rasters = [fitz.Rect(im["bbox"]) for im in page.get_image_info()]
    grects = list(rasters)
    for d in page.get_drawings():
        dr = fitz.Rect(d["rect"])
        if dr.width > 8 and dr.height > 8:
            grects.append(dr)

    figs = []
    for cr, ct in caps:
        cx0, cx1 = col_of(cr)

        def in_col(r: fitz.Rect) -> bool:  # x 与本栏有效重叠（双栏时隔离另一栏的内容）
            return min(r.x1, cx1) - max(r.x0, cx0) > (cx1 - cx0) * 0.15

        up = max([r.y1 for r in bounds if r.y1 <= cr.y0 - 2 and in_col(r)] + [0.0])
        below = [r.y0 for r in bounds if r.y0 >= cr.y1 + 2 and in_col(r)]
        down = min(below) if below else H * 0.95

        def ink(lo: float, hi: float) -> float:
            if hi <= lo:
                return 0.0
            s = 0.0
            for g in grects:
                if g.y0 >= lo - 3 and g.y1 <= hi + 3 and in_col(g):
                    s += (min(g.x1, cx1) - max(g.x0, cx0)) * g.height
            for r, t, k in tagged:
                if k == "minor" and r.y0 >= lo and r.y1 <= hi and in_col(r):
                    s += r.width * r.height
            return s

        ink_above, ink_below = ink(up, cr.y0), ink(cr.y1, down)
        if max(ink_above, ink_below) < (cx1 - cx0) * H * 0.02:
            continue  # 图注两侧（本栏内）都无成片图形 → 跳过
        if ink_above >= ink_below:
            top, bot = up, cr.y1            # 图在注上方（图注随之纳入裁剪底部）
        else:
            top, bot = cr.y0, down          # 图在注下方（图注随之纳入裁剪顶部）
        for r, t, k in tagged:              # 纳入紧贴图底的"资料来源"行
            if k == "src" and in_col(r) and top - 3 <= r.y0 and abs(r.y0 - bot) < 50:
                bot = max(bot, r.y1)
        # x 收紧到本栏内实际图形的横向范围，去掉栏内留白
        gx = [g for g in grects if g.y0 >= top - 3 and g.y1 <= bot + 3 and in_col(g)]
        if gx:
            x0, x1 = max(cx0, min(g.x0 for g in gx) - 4), min(cx1, max(g.x1 for g in gx) + 4)
        else:
            x0, x1 = cx0, cx1
        reg = fitz.Rect(x0, max(0, top - 2), x1, min(H, bot + 2))
        if reg.height < 70 or reg.width < 100:
            continue
        figs.append({
            "caption": ct,
            "bbox": [round(reg.x0, 1), round(reg.y0, 1), round(reg.x1, 1), round(reg.y1, 1)],
            "raster": any(reg.y0 - 3 <= g.y0 and g.y1 <= reg.y1 + 3 and in_col(g) for g in rasters),
        })
    return figs


def main() -> None:
    manifest = json.loads((PARSED_DIR / "manifest.json").read_text(encoding="utf-8"))
    reports = sorted({info["source"] for info in manifest.values()})
    total, with_raster, pages_seen = 0, 0, 0
    with open(FIGURES_PATH, "w", encoding="utf-8") as f:
        for n, source in enumerate(reports, 1):
            pdf_path = PDF_DIR / f"{source}.pdf"
            if not pdf_path.exists():
                continue
            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"  打开失败 《{source[:30]}》: {e}")
                continue
            cnt = 0
            for pno in range(len(doc)):
                pages_seen += 1
                for fig in detect_figures(doc[pno]):
                    rec = {"source": source, "page": pno + 1, **fig}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
                    cnt += 1
                    if fig["raster"]:
                        with_raster += 1
            doc.close()
            print(f"[{n}/{len(reports)}] 《{source[:34]}》 提取图 {cnt}")
    print(f"\n完成：{len(reports)}份报告 / {pages_seen}页 → 图注图 {total} 个"
          f"（含raster {with_raster}，矢量 {total - with_raster}）→ {FIGURES_PATH}")


if __name__ == "__main__":
    main()
