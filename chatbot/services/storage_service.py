"""
Storage backend dispatcher.

Picks between S3 and local-filesystem storage based on STORAGE_BACKEND (config/storage.py).
s3_service is only imported - and its boto3 client only created - when S3 is actually enabled,
so missing AWS credentials no longer break the app when running with local storage.
"""
from typing import Dict
from config.storage import S3_ENABLED
from utils.logger import get_logger

logger = get_logger(__name__)

if S3_ENABLED:
    from services.s3_service import s3_service as storage_service
    logger.info("Storage backend: S3")
else:
    from services.local_storage_service import local_storage_service as storage_service
    logger.info("Storage backend: local filesystem")


async def generate_presigned_url(file_name: str, file_type: str, file_size: int, folder: str = "uploads") -> Dict[str, str]:
    return await storage_service.generate_presigned_url(file_name, file_type, file_size, folder)


async def upload_payment_proof(order_id: int, file_name: str, file_type: str, file_size: int, file_content: bytes) -> Dict[str, str]:
    return await storage_service.upload_payment_proof_and_update_order(order_id, file_name, file_type, file_size, file_content)
