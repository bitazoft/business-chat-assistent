

"""
WhatsApp Cloud API Service
Handles sending and receiving messages through WhatsApp Business Cloud API for multiple accounts
"""

import os
import json
import requests
import re
from typing import Dict, Any, Optional, List
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
    media_id: Optional[str] = None
    media_mime_type: Optional[str] = None

@dataclass
class WhatsAppConfig:
    """Structure to hold configuration for a single WhatsApp account"""
    phone_number_id: str
    access_token: str
    verify_token: str
    business_account_id: Optional[str] = None
    base_url: str = None

    def __post_init__(self):
        if self.phone_number_id and not self.base_url:
            self.base_url = f"https://graph.facebook.com/v22.0/{self.phone_number_id}"

class WhatsAppService:
    """Service class for WhatsApp Cloud API integration with multiple accounts"""
    
    def __init__(self, validate_on_init: bool = True):
        self.configs: Dict[str, WhatsAppConfig] = {}
        self._load_configs()
        
        if validate_on_init:
            self._validate_configs()
            logger.info("✅ WhatsApp Service initialized successfully with %d accounts", len(self.configs))
        else:
            logger.info("⚠️ WhatsApp Service initialized without validation - call validate_configs() before use")
    
    def _load_configs(self):
        """Load all configurations from environment variables"""
        prefixes = set()
        # Find all configuration indices
        for key in os.environ:
            match = re.match(r'^WHATSAPP_CONFIG_(\d+)_', key)
            if match:
                prefixes.add(match.group(1))
        
        # Load configurations for each index
        for index in sorted(prefixes):
            phone_number_id = os.getenv(f'WHATSAPP_CONFIG_{index}_PHONE_NUMBER_ID')
            access_token = os.getenv(f'WHATSAPP_CONFIG_{index}_ACCESS_TOKEN')
            verify_token = os.getenv(f'WHATSAPP_CONFIG_{index}_VERIFY_TOKEN')
            business_account_id = os.getenv(f'WHATSAPP_CONFIG_{index}_BUSINESS_ACCOUNT_ID')
            
            if phone_number_id and access_token and verify_token:
                config = WhatsAppConfig(
                    phone_number_id=phone_number_id,
                    access_token=access_token,
                    verify_token=verify_token,
                    business_account_id=business_account_id
                )
                self.configs[phone_number_id] = config
            else:
                logger.warning("Incomplete configuration for WHATSAPP_CONFIG_%s", index)
    
    def _validate_configs(self):
        """Validate that all configurations have required fields"""
        invalid_configs = []
        for phone_number_id, config in self.configs.items():
            required_vars = [
                ("WHATSAPP_CONFIG_X_PHONE_NUMBER_ID", config.phone_number_id),
                ("WHATSAPP_CONFIG_X_ACCESS_TOKEN", config.access_token),
                ("WHATSAPP_CONFIG_X_VERIFY_TOKEN", config.verify_token)
            ]
            missing_vars = [var_name for var_name, var_value in required_vars if not var_value]
            if missing_vars:
                invalid_configs.append((phone_number_id, missing_vars))
        
        if invalid_configs:
            error_msg = "\n".join(
                f"Missing required variables for account {phone_id}: {', '.join(vars)}"
                for phone_id, vars in invalid_configs
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def validate_configs(self) -> bool:
        """
        Public method to validate all configurations
        
        Returns:
            True if all configurations are valid, False otherwise
        """
        try:
            self._validate_configs()
            return True
        except ValueError:
            return False
    
    def is_configured(self, phone_number_id: Optional[str] = None) -> bool:
        """
        Check if the service is properly configured for a specific account or any account
        
        Args:
            phone_number_id: Specific account to check (optional)
        
        Returns:
            True if the specified account or at least one account is configured
        """
        if phone_number_id:
            config = self.configs.get(phone_number_id)
            return bool(config and config.phone_number_id and config.access_token and config.verify_token)
        return bool(self.configs)
    
    def get_config(self, phone_number_id: str) -> Optional[WhatsAppConfig]:
        """
        Get configuration for a specific phone number ID
        
        Args:
            phone_number_id: Phone number ID of the account
            
        Returns:
            WhatsAppConfig object or None if not found
        """
        return self.configs.get(phone_number_id)
    
    def send_text_message(self, to_number: str, message: str, phone_number_id: str) -> Dict[str, Any]:
        """
        Send a text message via WhatsApp using a specific account
        
        Args:
            to_number: Recipient phone number (with country code, without +)
            message: Text message to send
            phone_number_id: Phone number ID of the sending account
            
        Returns:
            API response dictionary
        """
        config = self.get_config(phone_number_id)
        if not config or not self.is_configured(phone_number_id):
            error_msg = f"WhatsApp service is not properly configured for account {phone_number_id}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
        
        try:
            url = f"{config.base_url}/messages"
            
            headers = {
                "Authorization": f"Bearer {config.access_token}",
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
            
            logger.info(f"Sending WhatsApp message to {to_number} from account {phone_number_id}: {message[:50]}...")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Message sent successfully from {phone_number_id}. Message ID: {result.get('messages', [{}])[0].get('id', 'Unknown')}")
            
            return {
                "success": True,
                "message_id": result.get('messages', [{}])[0].get('id'),
                "response": result
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to send WhatsApp message from {phone_number_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
        except Exception as e:
            logger.error(f"❌ Unexpected error sending WhatsApp message from {phone_number_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def send_image_message(self, to_number: str, image_url: str, caption: str = "", phone_number_id: str = None) -> Dict[str, Any]:
        """
        Send an image message via WhatsApp using a specific account
        
        Args:
            to_number: Recipient phone number
            image_url: URL of the image to send
            caption: Optional caption for the image
            phone_number_id: Phone number ID of the sending account
            
        Returns:
            API response dictionary
        """
        config = self.get_config(phone_number_id)
        if not config or not self.is_configured(phone_number_id):
            error_msg = f"WhatsApp service is not properly configured for account {phone_number_id}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "response": None
            }
        
        try:
            url = f"{config.base_url}/messages"
            
            headers = {
                "Authorization": f"Bearer {config.access_token}",
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
            
            logger.info(f"Sending WhatsApp image to {to_number} from account {phone_number_id}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Image sent successfully from {phone_number_id}. Message ID: {result.get('messages', [{}])[0].get('id', 'Unknown')}")
            
            return {
                "success": True,
                "message_id": result.get('messages', [{}])[0].get('id'),
                "response": result
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to send WhatsApp image from {phone_number_id}: {str(e)}")
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
                logger.info("No messages found in webhook - might be status update")
                return None
            
            message_data = messages[0]
            
            from_number = message_data.get("from", "")
            message_id = message_data.get("id", "")
            timestamp = message_data.get("timestamp", "")
            message_type = message_data.get("type", "")
            
            content = ""
            media_id = None
            media_mime_type = None
            
            if message_type == "text":
                content = message_data.get("text", {}).get("body", "")
            elif message_type == "image":
                image_data = message_data.get("image", {})
                content = image_data.get("caption", "Image received")
                media_id = image_data.get("id")
                media_mime_type = image_data.get("mime_type")
            elif message_type == "audio":
                audio_data = message_data.get("audio", {})
                content = "[Audio message]"
                media_id = audio_data.get("id")
                media_mime_type = audio_data.get("mime_type")
            elif message_type == "video":
                video_data = message_data.get("video", {})
                content = video_data.get("caption", "[Video message]")
                media_id = video_data.get("id")
                media_mime_type = video_data.get("mime_type")
            elif message_type == "document":
                document_data = message_data.get("document", {})
                content = document_data.get("filename", "[Document]")
                media_id = document_data.get("id")
                media_mime_type = document_data.get("mime_type")
            else:
                content = f"[{message_type} message]"
            
            phone_number_id = value.get("metadata", {}).get("phone_number_id", "")
            
            logger.info(f"📨 Received WhatsApp message from {from_number} to account {phone_number_id}: {content[:50]} message Type : {message_type}")
            
            return WhatsAppMessage(
                from_number=from_number,
                to_number=phone_number_id,
                message_type=MessageType(message_type) if message_type in [mt.value for mt in MessageType] else MessageType.TEXT,
                content=content,
                message_id=message_id,
                timestamp=timestamp,
                media_id=media_id,
                media_mime_type=media_mime_type
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to parse webhook message: {str(e)}")
            return None
    
    def verify_webhook(self, verify_token: str, challenge: str) -> Optional[str]:
        """
        Verify webhook subscription for a specific account
        
        Args:
            verify_token: Token sent by WhatsApp
            challenge: Challenge string to echo back
            phone_number_id: Phone number ID of the account
            
        Returns:
            Challenge string if verification successful, None otherwise
        """
        for phone_number_id, config in self.configs.items():
            if config.verify_token == verify_token:
                logger.info(f"✅ Webhook verification successful for account {phone_number_id}")
                return challenge
        logger.warning("❌ Webhook verification failed - no matching token found")
        return None
    
    def mark_message_as_read(self, message_id: str, phone_number_id: str) -> bool:
        """
        Mark a message as read for a specific account
        
        Args:
            message_id: ID of the message to mark as read
            phone_number_id: Phone number ID of the account
            
        Returns:
            True if successful, False otherwise
        """
        config = self.get_config(phone_number_id)
        if not config or not self.is_configured(phone_number_id):
            logger.error(f"WhatsApp service is not properly configured for account {phone_number_id}")
            return False
        
        try:
            url = f"{config.base_url}/messages"
            
            headers = {
                "Authorization": f"Bearer {config.access_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info(f"✅ Message {message_id} marked as read for account {phone_number_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to mark message as read for account {phone_number_id}: {str(e)}")
            return False
    
    def download_image(self, whatsapp_message: WhatsAppMessage, save_directory: str = "./downloads") -> Dict[str, Any]:
        """
        Convenience method to download an image from a WhatsApp message
        
        Args:
            whatsapp_message: WhatsAppMessage object containing media info
            save_directory: Directory to save the image
            
        Returns:
            Dictionary with download result including file path
        """
        if whatsapp_message.message_type != MessageType.IMAGE or not whatsapp_message.media_id:
            return {
                "success": False,
                "error": "Message is not an image or missing media ID",
                "content": None,
                "file_path": None
            }
        
        # Generate filename based on media ID and mime type
        import os
        from datetime import datetime
        
        file_extension = ""
        if whatsapp_message.media_mime_type:
            if "jpeg" in whatsapp_message.media_mime_type or "jpg" in whatsapp_message.media_mime_type:
                file_extension = ".jpg"
            elif "png" in whatsapp_message.media_mime_type:
                file_extension = ".png"
            elif "gif" in whatsapp_message.media_mime_type:
                file_extension = ".gif"
            elif "webp" in whatsapp_message.media_mime_type:
                file_extension = ".webp"
        
        if not file_extension:
            file_extension = ".jpg"  # Default
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"whatsapp_image_{whatsapp_message.from_number}_{timestamp}_{whatsapp_message.media_id[:8]}{file_extension}"
        save_path = os.path.join(save_directory, filename)
        
        return self.download_media(
            media_id=whatsapp_message.media_id,
            phone_number_id=whatsapp_message.to_number,
            save_path=save_path
        )

    def download_media(self, media_id: str, phone_number_id: str, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Download media file from WhatsApp using media ID
        
        Args:
            media_id: Media ID from the message
            phone_number_id: Phone number ID of the account
            save_path: Optional path to save the file (if None, returns content in memory)
            
        Returns:
            Dictionary with download result
        """
        config = self.get_config(phone_number_id)
        if not config or not self.is_configured(phone_number_id):
            error_msg = f"WhatsApp service is not properly configured for account {phone_number_id}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "content": None,
                "file_path": None
            }
        
        try:
            # Step 1: Get media URL
            media_url = f"https://graph.facebook.com/v22.0/{media_id}"
            headers = {
                "Authorization": f"Bearer {config.access_token}"
            }
            
            logger.info(f"Getting media URL for {media_id} from account {phone_number_id}")
            
            response = requests.get(media_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            media_info = response.json()
            download_url = media_info.get("url")
            mime_type = media_info.get("mime_type", "")
            file_size = media_info.get("file_size", 0)
            
            if not download_url:
                error_msg = "No download URL found in media info"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "content": None,
                    "file_path": None
                }
            
            # Step 2: Download the actual file
            logger.info(f"Downloading media from {download_url}")
            
            download_response = requests.get(download_url, headers=headers, timeout=60)
            download_response.raise_for_status()
            
            content = download_response.content
            
            result = {
                "success": True,
                "mime_type": mime_type,
                "file_size": file_size,
                "media_id": media_id,
                "file_path": None
            }
            
            # Step 3: Save to file if path provided
            if save_path:
                import os
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                with open(save_path, 'wb') as f:
                    f.write(content)
                
                result["file_path"] = save_path
                logger.info(f"✅ Media saved to {save_path}")
            else:
                logger.info(f"✅ Media downloaded in memory ({len(content)} bytes)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to download media {media_id} from account {phone_number_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": None,
                "file_path": None
            }

    def get_profile_info(self, phone_number: str, phone_number_id: str) -> Dict[str, Any]:
        """
        Get profile information for a WhatsApp user using a specific account
        
        Args:
            phone_number: User's phone number
            phone_number_id: Phone number ID of the account
            
        Returns:
            Profile information dictionary
        """
        config = self.get_config(phone_number_id)
        if not config or not self.is_configured(phone_number_id):
            logger.error(f"WhatsApp service is not properly configured for account {phone_number_id}")
            return {
                "success": False,
                "error": f"No configuration for account {phone_number_id}",
                "profile": None
            }
        
        try:
            url = f"https://graph.facebook.com/v22.0/{phone_number}"
            
            headers = {
                "Authorization": f"Bearer {config.access_token}"
            }
            
            params = {
                "fields": "profile_name"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Retrieved profile for {phone_number} using account {phone_number_id}")
            
            return {
                "success": True,
                "profile": result
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get profile info for account {phone_number_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "profile": None
            }
    
    def get_seller_id(self, phone_number_id: str) -> Optional[str]:
        """
        Get seller ID associated with a WhatsApp number
        
        Args:
            phone_number_id: Phone number ID of the account
            
        Returns:
            Seller ID if found, None otherwise
        """
        config = self.get_config(phone_number_id)
        if not config:
            logger.warning(f"❌ No configuration found for account {phone_number_id}")
            return None
        
        logger.info(f"Retrieving seller ID for {phone_number_id}")
        return config.business_account_id or "default_seller"

# Global instance - initialized without validation to prevent import errors
# Call validate_configs() or check is_configured() before using
whatsapp_service = WhatsAppService(validate_on_init=False)