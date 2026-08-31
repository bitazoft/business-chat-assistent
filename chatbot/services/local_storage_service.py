import os
import time
from typing import Dict
from utils.file_validator import validate_file, validate_file_extension
from db.database import SessionLocal
from models.schemas import Order
from sqlalchemy.exc import SQLAlchemyError
from config.storage import LOCAL_STORAGE_DIR, LOCAL_STORAGE_BASE_URL
from utils.logger import get_logger

logger = get_logger(__name__)


class LocalStorageService:
    """Filesystem-backed drop-in replacement for S3Service, used when STORAGE_BACKEND=local."""

    def __init__(self):
        self.base_dir = LOCAL_STORAGE_DIR
        self.base_url = LOCAL_STORAGE_BASE_URL
        os.makedirs(self.base_dir, exist_ok=True)

    def _full_path(self, key: str) -> str:
        return os.path.join(self.base_dir, *key.split("/"))

    def _file_url(self, key: str) -> str:
        return f"{self.base_url}/{key}"

    def _write(self, key: str, content: bytes) -> None:
        full_path = self._full_path(key)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(content)

    async def generate_presigned_url(self, file_name: str, file_type: str, file_size: int, folder: str = "uploads") -> Dict[str, str]:
        """No presign concept locally - just hand back the key/URL the caller should write the file to."""
        validate_file(file_type, file_size)
        validate_file_extension(file_name, file_type)

        timestamp = int(time.time() * 1000)
        unique_file_name = f"{folder}/{timestamp}-{file_name}"

        return {
            "upload_url": self._file_url(unique_file_name),
            "file_url": self._file_url(unique_file_name),
            "key": unique_file_name,
        }

    async def upload_payment_proof_and_update_order(self, order_id: int, file_name: str, file_type: str, file_size: int, file_content: bytes) -> Dict[str, str]:
        db = SessionLocal()
        saved_key = None

        try:
            validate_file(file_type, file_size)
            validate_file_extension(file_name, file_type)

            order = db.query(Order).filter(Order.id == order_id).first()
            if not order:
                raise ValueError(f"Order with ID {order_id} not found")

            timestamp = int(time.time() * 1000)
            saved_key = f"payment-proofs/{timestamp}-{file_name}"
            self._write(saved_key, file_content)

            file_url = self._file_url(saved_key)
            order.payment_proof = file_url
            db.commit()

            # Mirrors S3Service behaviour: the stored copy is removed right after the URL is saved.
            await self.delete_file(saved_key)

            return {
                "success": True,
                "message": "Payment proof uploaded and order updated successfully. Image has been deleted.",
                "file_url": file_url,
                "order_id": order_id,
            }

        except (ValueError, SQLAlchemyError) as e:
            db.rollback()
            if saved_key:
                try:
                    await self.delete_file(saved_key)
                except Exception:
                    pass
            raise Exception(f"Failed to process payment proof: {str(e)}")
        finally:
            db.close()

    async def delete_file(self, file_key: str) -> bool:
        path = self._full_path(file_key)
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except OSError as e:
            logger.error(f"Error deleting local file {file_key}: {str(e)}")
            return False

    async def upload_file_direct(self, file_name: str, file_type: str, file_content: bytes, folder: str = "uploads") -> Dict[str, str]:
        validate_file(file_type, len(file_content))
        validate_file_extension(file_name, file_type)

        timestamp = int(time.time() * 1000)
        unique_file_name = f"{folder}/{timestamp}-{file_name}"
        self._write(unique_file_name, file_content)

        return {
            "file_url": self._file_url(unique_file_name),
            "key": unique_file_name,
        }


local_storage_service = LocalStorageService()
