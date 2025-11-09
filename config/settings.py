from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

SHL_BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RECOMMENDATIONS = 10

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
