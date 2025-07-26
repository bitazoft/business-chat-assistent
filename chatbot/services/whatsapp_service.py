"""
WhatsApp Cloud API Service
Handles sending and receiving messages through WhatsApp Business Cloud API
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from utils.logger import get_logger
from dataclasses import dataclass
from enum import Enum

logger = get_logger(__name__)

class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"

@dataclass
class WhatsAppMessage:
    """WhatsApp message structure"""
    from_number: str
    to_number: str
    message_type: MessageType
    content: str
    message_id: Optional[str] = None
    timestamp: Optional[str] = None

class WhatsAppService:
    """Service class for WhatsApp Cloud API integration"""
    
    def __init__(self, validate_on_init: bool = True):
        self.access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
        self.phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        self.verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN")
        self.business_account_id = os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID")
        
        # WhatsApp API base URL - will be set properly when config is validated
        self.base_url = None
        if self.phone_number_id:
            self.base_url = f"https://graph.facebook.com/v22.0/{self.phone_number_id}"
        
        # Validate required environment variables if requested
        if validate_on_init:
            self._validate_config()
            logger.info("✅ WhatsApp Service initialized successfully")
        else:
            logger.info("⚠️ WhatsApp Service initialized without validation - call validate_config() before use")
    
    def _validate_config(self):
        """Validate that all required environment variables are set"""
        required_vars = [
            ("WHATSAPP_ACCESS_TOKEN", self.access_token),
            ("WHATSAPP_PHONE_NUMBER_ID", self.phone_number_id),
            ("WHATSAPP_VERIFY_TOKEN", self.verify_token)
        ]
        
        missing_vars = [var_name for var_name, var_value in required_vars if not var_value]
        
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Update base URL now that we have valid phone_number_id
        if not self.base_url:
            self.base_url = f"https://graph.facebook.com/v22.0/{self.phone_number_id}"
    
    def validate_config(self) -> bool:
        """
        Public method to validate configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            self._validate_config()
            return True
        except ValueError:
            return False
    
    def is_configured(self) -> bool:
        """
        Check if the service is properly configured
        
        Returns:
            True if all required environment variables are set
        """
        return all([
            self.access_token,
            self.phone_number_id,
            self.verify_token
        ])
    
    def send_text_message(self, to_number: str, message: str) -> Dict[str, Any]:
        """
        Send a text message via WhatsApp
        
        Args:
            to_number: Recipient phone number (with country code, without +)
            message: Text message to send
            
        Returns:
            API response dictionary
        """
        # Validate configuration before proceeding
        if not self.is_configured():
            error_msg = "WhatsApp service is not properly configured. Missing environment variables."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
        
        try:
            url = f"{self.base_url}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "text",
                "text": {
                    "body": message
                }
            }
            
            logger.info(f"Sending WhatsApp message to {to_number}: {message[:50]}...")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Message sent successfully. Message ID: {result.get('messages', [{}])[0].get('id', 'Unknown')}")
            
            return {
                "success": True,
                "message_id": result.get('messages', [{}])[0].get('id'),
                "response": result
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to send WhatsApp message: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error sending WhatsApp message: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def send_image_message(self, to_number: str, image_url: str, caption: str = "") -> Dict[str, Any]:
        """
        Send an image message via WhatsApp
        
        Args:
            to_number: Recipient phone number
            image_url: URL of the image to send
            caption: Optional caption for the image
            
        Returns:
            API response dictionary
        """
        # Validate configuration before proceeding
        if not self.is_configured():
            error_msg = "WhatsApp service is not properly configured. Missing environment variables."
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
        
        try:
            url = f"{self.base_url}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "to": to_number,
                "type": "image",
                "image": {
                    "link": image_url,
                    "caption": caption
                }
            }
            
            logger.info(f"Sending WhatsApp image to {to_number}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Image sent successfully. Message ID: {result.get('messages', [{}])[0].get('id', 'Unknown')}")
            
            return {
                "success": True,
                "message_id": result.get('messages', [{}])[0].get('id'),
                "response": result
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to send WhatsApp image: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def parse_webhook_message(self, webhook_data: Dict[str, Any]) -> Optional[WhatsAppMessage]:
        """
        Parse incoming webhook data to extract message information
        
        Args:
            webhook_data: Raw webhook data from WhatsApp
            
        Returns:
            WhatsAppMessage object or None if parsing fails
        """
        try:
            # Navigate through the webhook structure
            entry = webhook_data.get("entry", [])
            if not entry:
                logger.warning("No entry found in webhook data")
                return None
            
            changes = entry[0].get("changes", [])
            if not changes:
                logger.warning("No changes found in webhook data")
                return None
            
            value = changes[0].get("value", {})
            messages = value.get("messages", [])
            
            if not messages:
                # This might be a status update, not a message
                logger.info("No messages found in webhook - might be status update")
                return None
            
            message_data = messages[0]
            
            # Extract message details
            from_number = message_data.get("from", "")
            message_id = message_data.get("id", "")
            timestamp = message_data.get("timestamp", "")
            message_type = message_data.get("type", "")
            
            # Extract message content based on type
            content = ""
            if message_type == "text":
                content = message_data.get("text", {}).get("body", "")
            elif message_type == "image":
                content = message_data.get("image", {}).get("caption", "")
            elif message_type == "audio":
                content = "[Audio message]"
            elif message_type == "video":
                content = message_data.get("video", {}).get("caption", "[Video message]")
            elif message_type == "document":
                content = message_data.get("document", {}).get("filename", "[Document]")
            else:
                content = f"[{message_type} message]"
            
            # Get phone number ID (your WhatsApp number)
            phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
            
            logger.info(f"📨 Received WhatsApp message from {from_number}: {content[:50]}...")
            
            return WhatsAppMessage(
                from_number=from_number,
                to_number=phone_number_id,
                message_type=MessageType(message_type) if message_type in [mt.value for mt in MessageType] else MessageType.TEXT,
                content=content,
                message_id=message_id,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse webhook message: {str(e)}")
            return None
    
    def verify_webhook(self, verify_token: str, challenge: str) -> Optional[str]:
        """
        Verify webhook subscription
        
        Args:
            verify_token: Token sent by WhatsApp
            challenge: Challenge string to echo back
            
        Returns:
            Challenge string if verification successful, None otherwise
        """
        if verify_token == self.verify_token:
            logger.info("✅ Webhook verification successful")
            return challenge
        else:
            logger.warning("❌ Webhook verification failed - invalid token")
            return None
    
    def mark_message_as_read(self, message_id: str) -> bool:
        """
        Mark a message as read
        
        Args:
            message_id: ID of the message to mark as read
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/messages"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"✅ Message {message_id} marked as read")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to mark message as read: {str(e)}")
            return False
    
    def get_profile_info(self, phone_number: str) -> Dict[str, Any]:
        """
        Get profile information for a WhatsApp user
        
        Args:
            phone_number: User's phone number
            
        Returns:
            Profile information dictionary
        """
        try:
            url = f"https://graph.facebook.com/v22.0/{phone_number}"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}"
            }
            
            params = {
                "fields": "profile_name"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Retrieved profile for {phone_number}")
            
            return {
                "success": True,
                "profile": result
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get profile info: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "profile": None
            }

    def get_seller_id(self, phone_number_id: str) -> Optional[str]:
        """
        Get seller ID associated with a WhatsApp number
        
        Args:
            phone_number: User's phone number
            
        Returns:
            Seller ID if found, None otherwise
        """
        
        
        logger.info(f"Retrieving seller ID for {phone_number_id}")
        return "default_seller"

# Global instance - initialized without validation to prevent import errors
# Call validate_config() or check is_configured() before using
whatsapp_service = WhatsAppService(validate_on_init=False)
