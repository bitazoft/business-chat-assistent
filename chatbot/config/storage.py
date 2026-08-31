# Storage backend configuration
# STORAGE_BACKEND=s3    -> payment proofs / uploads go to AWS S3 (config/aws_s3.py, services/s3_service.py)
# STORAGE_BACKEND=local -> payment proofs / uploads are saved to a local folder instead (default)
import os
from dotenv import load_dotenv

load_dotenv()

STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").strip().lower()
S3_ENABLED = STORAGE_BACKEND == "s3"

# Only used when STORAGE_BACKEND=local
LOCAL_STORAGE_DIR = os.getenv("LOCAL_STORAGE_DIR", "uploads")
LOCAL_STORAGE_BASE_URL = os.getenv("LOCAL_STORAGE_BASE_URL", "http://localhost:8001/uploads").rstrip("/")
