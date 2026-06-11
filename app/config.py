from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "具身智能"
DATA_DIR = PROJECT_ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_DEVICE = "cuda"
EMBEDDING_BATCH_SIZE = 16

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

TOP_K = 6
