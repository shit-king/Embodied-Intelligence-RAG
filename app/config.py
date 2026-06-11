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
# 问答模型双档：快速/思考，由前端切换；agent内部决策（路由/拆解/改写）固定走快速档
DEEPSEEK_MODELS = {
    "fast": "deepseek-v4-flash",
    "thinking": "deepseek-v4-pro",
}
DEFAULT_MODE = "fast"

VL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
VL_MODEL = "qwen-vl-plus"

TOP_K = 6
