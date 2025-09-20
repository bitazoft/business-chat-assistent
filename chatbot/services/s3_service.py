import boto3
import os
import time
from typing import Dict, Optional
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv
from utils.file_validator import validate_file, validate_file_extension
from db.database import SessionLocal
from models.schemas import Order
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

class S3Service:
    def __init__(self):
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'us-east-1')
        self.bucket_name = os.getenv('AWS_BUCKET_NAME')
        
        if not all([self.aws_access_key_id, self.aws_secret_access_key, self.bucket_name]):
            logger.error("Missing required AWS credentials or bucket name in environment variables")
            raise ValueError("Missing required AWS credentials or bucket name in environment variables")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.aws_region
        )
    
    async def generate_presigned_url(self, file_name: str, file_type: str, file_size: int, folder: str = "uploads") -> Dict[str, str]:
        """
        Generate a presigned URL for uploading files to S3
        
        Args:
            file_name (str): Name of the file
            file_type (str): MIME type of the file
            file_size (int): Size of the file in bytes
            folder (str): Folder path in S3 bucket
            
        Returns:
            Dict[str, str]: Dictionary containing upload_url and file_url
            
        Raises:
            ValueError: If file validation fails
            ClientError: If AWS S3 operation fails
        """
        try:
            # Validate file
            validate_file(file_type, file_size)
            validate_file_extension(file_name, file_type)
            
            # Generate unique filename with timestamp
            timestamp = int(time.time() * 1000)  # milliseconds
            unique_file_name = f"{folder}/{timestamp}-{file_name}"
            
            # S3 upload parameters
            params = {
                'Bucket': self.bucket_name,
                'Key': unique_file_name,
                'ContentType': file_type,
            }
            
            # Generate presigned URL for upload
            upload_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params=params,
                ExpiresIn=300  # 5 minutes
            )
            
            # Generate the final file URL
            file_url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{unique_file_name}"
            
            return {
                'upload_url': upload_url,
                'file_url': file_url,
                'key': unique_file_name
            }
            
        except (ValueError, ClientError) as e:
            raise Exception(f"Could not generate presigned URL: {str(e)}")
    
    async def upload_payment_proof_and_update_order(self, order_id: int, file_name: str, file_type: str, file_size: int, file_content: bytes) -> Dict[str, str]:
        """
        Upload payment proof image and update order with the file URL, then delete the image
        
        Args:
            order_id (int): ID of the order to update
            file_name (str): Name of the payment proof file
            file_type (str): MIME type of the file
            file_size (int): Size of the file in bytes
            file_content (bytes): File content as bytes
            
        Returns:
            Dict[str, str]: Result with success status and file URL
            
        Raises:
            Exception: If any operation fails
        """
        db = SessionLocal()
        uploaded_key = None
        
        try:
            # Validate file
            validate_file(file_type, file_size)
            validate_file_extension(file_name, file_type)
            
            # Check if order exists
            order = db.query(Order).filter(Order.id == order_id).first()
            if not order:
                raise ValueError(f"Order with ID {order_id} not found")
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            unique_file_name = f"payment-proofs/{timestamp}-{file_name}"
            uploaded_key = unique_file_name
            
            # Upload file to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=unique_file_name,
                Body=file_content,
                ContentType=file_type
            )
            
            # Generate file URL
            file_url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{unique_file_name}"
            
            # Update order with payment proof URL
            order.payment_proof = file_url
            db.commit()
            
            # Delete the uploaded image after updating the order
            await self.delete_file(unique_file_name)
            
            return {
                'success': True,
                'message': 'Payment proof uploaded and order updated successfully. Image has been deleted.',
                'file_url': file_url,
                'order_id': order_id
            }
            
        except (ValueError, ClientError, SQLAlchemyError) as e:
            db.rollback()
            # If upload succeeded but DB update failed, clean up the uploaded file
            if uploaded_key:
                try:
                    await self.delete_file(uploaded_key)
                except:
                    pass  # Ignore cleanup errors
            raise Exception(f"Failed to process payment proof: {str(e)}")
        finally:
            db.close()
    
    async def delete_file(self, file_key: str) -> bool:
        """
        Delete a file from S3
        
        Args:
            file_key (str): S3 object key to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_key
            )
            return True
        except ClientError as e:
            print(f"Error deleting file {file_key}: {str(e)}")
            return False
    
    async def upload_file_direct(self, file_name: str, file_type: str, file_content: bytes, folder: str = "uploads") -> Dict[str, str]:
        """
        Upload file directly to S3 (without deleting afterwards)
        
        Args:
            file_name (str): Name of the file
            file_type (str): MIME type of the file
            file_content (bytes): File content as bytes
            folder (str): Folder path in S3 bucket
            
        Returns:
            Dict[str, str]: Dictionary containing file_url and key
        """
        try:
            # Validate file
            validate_file(file_type, len(file_content))
            validate_file_extension(file_name, file_type)
            
            # Generate unique filename
            timestamp = int(time.time() * 1000)
            unique_file_name = f"{folder}/{timestamp}-{file_name}"
            
            # Upload file to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=unique_file_name,
                Body=file_content,
                ContentType=file_type
            )
            
            # Generate file URL
            file_url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{unique_file_name}"
            
            return {
                'file_url': file_url,
                'key': unique_file_name
            }
            
        except (ValueError, ClientError) as e:
            raise Exception(f"Could not upload file: {str(e)}")

# Create a singleton instance
s3_service = S3Service()

# Export functions for backward compatibility
async def generate_presigned_url(file_name: str, file_type: str, file_size: int, folder: str = "uploads") -> Dict[str, str]:
    """Generate presigned URL for file upload"""
    return await s3_service.generate_presigned_url(file_name, file_type, file_size, folder)

async def upload_payment_proof(order_id: int, file_name: str, file_type: str, file_size: int, file_content: bytes) -> Dict[str, str]:
    """Upload payment proof and update order, then delete the image"""
    return await s3_service.upload_payment_proof_and_update_order(order_id, file_name, file_type, file_size, file_content)
