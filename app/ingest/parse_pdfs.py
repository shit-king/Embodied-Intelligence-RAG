"""批量解析PDF：按页提取文本，内容哈希去重，结果存JSON支持断点续跑。"""
import hashlib
import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.config import PARSED_DIR, PDF_DIR

MANIFEST_PATH = PARSED_DIR / "manifest.json"
MIN_PAGE_CHARS = 20


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_one(pdf_path: Path) -> tuple[list[dict], list[int]]:
    pages, empty_pages = [], []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            page_no = i + 1
            if len(text) < MIN_PAGE_CHARS:
                empty_pages.append(page_no)
            else:
                pages.append({"page": page_no, "text": text})
    return pages, empty_pages


def main() -> None:
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    known_hashes = set(manifest)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"发现 {len(pdf_files)} 个PDF，已解析 {len(known_hashes)} 个")

    for n, pdf_path in enumerate(pdf_files, 1):
        digest = file_md5(pdf_path)
        if digest in known_hashes:
            continue
        print(f"[{n}/{len(pdf_files)}] {pdf_path.name}")
        try:
            pages, empty_pages = parse_one(pdf_path)
        except Exception as e:
            print(f"  解析失败: {e}")
            manifest[digest] = {"source": pdf_path.stem, "error": str(e)}
            save_manifest(manifest)
            continue

        (PARSED_DIR / f"{digest}.json").write_text(
            json.dumps(
                {"source": pdf_path.stem, "pages": pages},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifest[digest] = {
            "source": pdf_path.stem,
            "text_pages": len(pages),
            "empty_pages": empty_pages,
        }
        known_hashes.add(digest)
        save_manifest(manifest)
        if empty_pages:
            print(f"  文本页 {len(pages)}，跳过扫描/图表页 {len(empty_pages)}")

    total_pages = sum(m.get("text_pages", 0) for m in manifest.values())
    total_empty = sum(len(m.get("empty_pages", [])) for m in manifest.values())
    errors = [m["source"] for m in manifest.values() if "error" in m]
    print(f"\n完成：{len(manifest)} 份文档（去重后），文本页 {total_pages}，跳过页 {total_empty}")
    if errors:
        print(f"解析失败 {len(errors)} 份: {errors}")


if __name__ == "__main__":
    main()
