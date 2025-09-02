#!/usr/bin/env python3
"""
Test cases for upload_payment_proof_and_update_order function
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import asyncio
from pathlib import Path

# Add the chatbot directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the function to test
from repositories.tools import upload_payment_proof_and_update_order


class TestUploadPaymentProofAndUpdateOrder(unittest.TestCase):
    """Test cases for upload_payment_proof_and_update_order function"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_order_id = 123
        self.test_file_content = b"fake_image_content"
        self.test_file_url = "https://s3.amazonaws.com/bucket/payment-proofs/test_image.jpg"
        
        # Create a temporary test file
        self.temp_dir = tempfile.mkdtemp()
        self.test_file_path = os.path.join(self.temp_dir, "test_payment_proof.jpg")
        with open(self.test_file_path, 'wb') as f:
            f.write(self.test_file_content)

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        os.rmdir(self.temp_dir)

    @patch('repositories.tools.SessionLocal')
    @patch('repositories.tools.asyncio.run')
    @patch('repositories.tools.os.path.exists')
    @patch('repositories.tools.os.remove')
    def test_successful_upload_and_update(self, mock_remove, mock_exists, mock_asyncio_run, mock_session_local):
        """Test successful upload and database update"""
        # Mock file exists
        mock_exists.return_value = True
        
        # Mock database session and order
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_order = Mock()
        mock_order.id = self.test_order_id
        mock_db.query.return_value.filter.return_value.first.return_value = mock_order
        
        # Mock S3 upload result
        mock_s3_result = {
            'file_url': self.test_file_url,
            'key': 'payment-proofs/test_payment_proof.jpg'
        }
        mock_asyncio_run.return_value = mock_s3_result
        
        # Mock S3 service import
        with patch('repositories.tools.s3_service') as mock_s3_service:
            mock_s3_service.upload_file_direct.return_value = mock_s3_result
            
            # Call the function
            result = upload_payment_proof_and_update_order(self.test_order_id, self.test_file_path)
            
            # Assertions
            self.assertIn("Payment proof uploaded successfully", result)
            self.assertIn(str(self.test_order_id), result)
            self.assertIn(self.test_file_url, result)
            self.assertIn("has been deleted", result)
            
            # Verify order was updated
            self.assertEqual(mock_order.payment_proof, self.test_file_url)
            mock_db.commit.assert_called_once()
            
            # Verify file was deleted
            mock_remove.assert_called_once_with(self.test_file_path)

    @patch('repositories.tools.os.path.exists')
    def test_file_not_found(self, mock_exists):
        """Test error when file doesn't exist"""
        mock_exists.return_value = False
        
        result = upload_payment_proof_and_update_order(self.test_order_id, "/nonexistent/file.jpg")
        
        self.assertIn("Error: File not found", result)
        self.assertIn("/nonexistent/file.jpg", result)

    @patch('repositories.tools.os.path.exists')
    def test_unsupported_file_type(self, mock_exists):
        """Test error for unsupported file types"""
        mock_exists.return_value = True
        unsupported_file = "/path/to/file.txt"
        
        result = upload_payment_proof_and_update_order(self.test_order_id, unsupported_file)
        
        self.assertIn("Error: Unsupported file type", result)
        self.assertIn(".txt", result)

    @patch('repositories.tools.SessionLocal')
    @patch('repositories.tools.os.path.exists')
    def test_order_not_found(self, mock_exists, mock_session_local):
        """Test error when order doesn't exist"""
        mock_exists.return_value = True
        
        # Mock database session
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        result = upload_payment_proof_and_update_order(999, self.test_file_path)
        
        self.assertIn("Error: Order with ID 999 not found", result)

    @patch('repositories.tools.os.path.exists')
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_file_read_error(self, mock_open, mock_exists):
        """Test error when file cannot be read"""
        mock_exists.return_value = True
        
        result = upload_payment_proof_and_update_order(self.test_order_id, self.test_file_path)
        
        self.assertIn("Error reading file", result)
        self.assertIn("Permission denied", result)

    @patch('repositories.tools.SessionLocal')
    @patch('repositories.tools.os.path.exists')
    def test_s3_service_not_available(self, mock_exists, mock_session_local):
        """Test error when S3 service is not available (boto3 not installed)"""
        mock_exists.return_value = True
        
        # Mock database session and order
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_order = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_order
        
        # Mock ImportError for S3 service
        with patch('repositories.tools.s3_service', side_effect=ImportError("No module named 'boto3'")):
            result = upload_payment_proof_and_update_order(self.test_order_id, self.test_file_path)
            
            self.assertIn("Error: S3 service not available", result)
            self.assertIn("boto3 not installed", result)

    @patch('repositories.tools.SessionLocal')
    @patch('repositories.tools.asyncio.run')
    @patch('repositories.tools.os.path.exists')
    def test_s3_upload_error(self, mock_exists, mock_asyncio_run, mock_session_local):
        """Test error during S3 upload"""
        mock_exists.return_value = True
        
        # Mock database session and order
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_order = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_order
        
        # Mock S3 upload error
        mock_asyncio_run.side_effect = Exception("S3 upload failed")
        
        with patch('repositories.tools.s3_service'):
            result = upload_payment_proof_and_update_order(self.test_order_id, self.test_file_path)
            
            self.assertIn("Error uploading to S3", result)
            self.assertIn("S3 upload failed", result)

    @patch('repositories.tools.SessionLocal')
    @patch('repositories.tools.asyncio.run')
    @patch('repositories.tools.os.path.exists')
    @patch('repositories.tools.os.remove')
    def test_database_update_error_with_s3_cleanup(self, mock_remove, mock_exists, mock_asyncio_run, mock_session_local):
        """Test database error with S3 cleanup"""
        mock_exists.return_value = True
        
        # Mock S3 upload result
        mock_s3_result = {
            'file_url': self.test_file_url,
            'key': 'payment-proofs/test_payment_proof.jpg'
        }
        mock_asyncio_run.return_value = mock_s3_result
        
        # Mock database session with error
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_order = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_order
        mock_db.commit.side_effect = Exception("Database error")
        
        with patch('repositories.tools.s3_service') as mock_s3_service:
            mock_s3_service.upload_file_direct.return_value = mock_s3_result
            mock_s3_service.delete_file.return_value = None
            
            result = upload_payment_proof_and_update_order(self.test_order_id, self.test_file_path)
            
            self.assertIn("Error updating order in database", result)
            self.assertIn("Database error", result)
            
            # Verify S3 cleanup was attempted
            mock_asyncio_run.assert_called()
            mock_db.rollback.assert_called_once()

    @patch('repositories.tools.SessionLocal')
    @patch('repositories.tools.asyncio.run')
    @patch('repositories.tools.os.path.exists')
    @patch('repositories.tools.os.remove', side_effect=OSError("File deletion failed"))
    def test_file_deletion_warning(self, mock_remove, mock_exists, mock_asyncio_run, mock_session_local):
        """Test warning when local file deletion fails"""
        mock_exists.return_value = True
        
        # Mock database session and order
        mock_db = Mock()
        mock_session_local.return_value = mock_db
        mock_order = Mock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_order
        
        # Mock S3 upload result
        mock_s3_result = {
            'file_url': self.test_file_url,
            'key': 'payment-proofs/test_payment_proof.jpg'
        }
        mock_asyncio_run.return_value = mock_s3_result
        
        with patch('repositories.tools.s3_service') as mock_s3_service:
            mock_s3_service.upload_file_direct.return_value = mock_s3_result
            
            result = upload_payment_proof_and_update_order(self.test_order_id, self.test_file_path)
            
            self.assertIn("Payment proof uploaded successfully", result)
            self.assertIn("Warning: Could not delete local file", result)
            self.assertIn("File deletion failed", result)

    def test_supported_file_types(self):
        """Test that all supported file types are correctly mapped"""
        # Create temporary files with different extensions
        supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf']
        
        for ext in supported_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
                temp_file.write(b"test content")
                temp_path = temp_file.name
            
            try:
                with patch('repositories.tools.os.path.exists', return_value=True), \
                     patch('repositories.tools.SessionLocal') as mock_session_local:
                    
                    # Mock database session and order not found to stop early
                    mock_db = Mock()
                    mock_session_local.return_value = mock_db
                    mock_db.query.return_value.filter.return_value.first.return_value = None
                    
                    result = upload_payment_proof_and_update_order(self.test_order_id, temp_path)
                    
                    # Should not get unsupported file type error
                    self.assertNotIn("Unsupported file type", result)
                    # Should get order not found error instead (meaning file type was accepted)
                    self.assertIn("Order with ID", result)
            finally:
                os.unlink(temp_path)

    @patch('repositories.tools.os.path.exists')
    @patch('repositories.tools.os.path.getsize')
    @patch('repositories.tools.os.path.basename')
    def test_file_info_extraction(self, mock_basename, mock_getsize, mock_exists):
        """Test correct extraction of file information"""
        mock_exists.return_value = True
        mock_basename.return_value = "payment_proof.jpg"
        mock_getsize.return_value = 1024
        
        with patch('repositories.tools.SessionLocal') as mock_session_local:
            mock_db = Mock()
            mock_session_local.return_value = mock_db
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            result = upload_payment_proof_and_update_order(self.test_order_id, "/path/to/payment_proof.jpg")
            
            # Verify file info functions were called
            mock_basename.assert_called_once_with("/path/to/payment_proof.jpg")
            mock_getsize.assert_called_once_with("/path/to/payment_proof.jpg")


def run_tests():
    """Run all test cases"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
