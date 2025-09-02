# """
# DeepSeek Vision Image Analysis Service
# Handles image analysis using DeepSeek's vision model
# """

# import os
# import base64
# import requests
# from typing import Dict, Any, Optional, List
# from utils.logger import get_logger
# import json
# import torch
# from transformers import AutoModelForCausalLM

# from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
# from deepseek_vl.utils.io import load_pil_images

# logger = get_logger(__name__)

# class DeepSeekVisionService:
#     # """Service for analyzing images using DeepSeek Vision model"""
    
#     # def __init__(self):
#     #     self.api_key = os.getenv("DEEPSEEK_API_KEY")
#     #     self.base_url = "https://api.deepseek.com/v1/chat/completions"
#     #     self.model = "deepseek-vl"
        
#     #     if not self.api_key:
#     #         logger.warning("⚠️ DEEPSEEK_API_KEY not found in environment variables")
    
#     # def encode_image_to_base64(self, image_path: str) -> str:
#     #     """
#     #     Encode image file to base64 string
        
#     #     Args:
#     #         image_path: Path to the image file
            
#     #     Returns:
#     #         Base64 encoded image string
#     #     """
#     #     try:
#     #         with open(image_path, "rb") as image_file:
#     #             return base64.b64encode(image_file.read()).decode('utf-8')
#     #     except Exception as e:
#     #         logger.error(f"❌ Error encoding image to base64: {str(e)}")
#     #         raise
    
#     # def analyze_image(self, image_path: str, prompt: str = None, language: str = "en") -> Dict[str, Any]:
#     #     """
#     #     Analyze an image using DeepSeek Vision
        
#     #     Args:
#     #         image_path: Path to the image file
#     #         prompt: Custom prompt for image analysis (optional)
#     #         language: Language for the response (en, ar, etc.)
            
#     #     Returns:
#     #         Dictionary containing analysis results
#     #     """
#     #     if not self.api_key:
#     #         return {
#     #             "success": False,
#     #             "error": "DeepSeek API key not configured",
#     #             "analysis": None
#     #         }
        
#     #     if not os.path.exists(image_path):
#     #         return {
#     #             "success": False,
#     #             "error": f"Image file not found: {image_path}",
#     #             "analysis": None
#     #         }
        
#     #     try:
#     #         # Encode image to base64
#     #         base64_image = self.encode_image_to_base64(image_path)
            
#     #         # Default prompt based on language
#     #         if not prompt:
#     #                 prompt = """Analyze this image carefully. Please describe:
#     #                 1. What you see in the image
#     #                 2. Main objects and important details
#     #                 3. Colors and composition
#     #                 4. Any text present in the image
#     #                 5. The context or likely purpose of the image

#     #                 Be descriptive and detailed in your response."""
            
#     #         # Prepare the request payload
#     #         payload = {
#     #             "model": self.model,
#     #             "messages": [
#     #                 {
#     #                     "role": "user",
#     #                     "content": [
#     #                         {
#     #                             "type": "text",
#     #                             "text": prompt
#     #                         },
#     #                         {
#     #                             "type": "image_url",
#     #                             "image_url": {
#     #                                 "url": f"data:image/jpeg;base64,{base64_image}"
#     #                             }
#     #                         }
#     #                     ]
#     #                 }
#     #             ],
#     #             "max_tokens": 1000,
#     #             "temperature": 0.7
#     #         }
            
#     #         headers = {
#     #             "Authorization": f"Bearer {self.api_key}",
#     #             "Content-Type": "application/json"
#     #         }
            
#     #         logger.info(f"🔍 Analyzing image with DeepSeek Vision: {image_path}")
            
#     #         # Make the API request
#     #         response = requests.post(
#     #             self.base_url,
#     #             headers=headers,
#     #             json=payload,
#     #             timeout=60
#     #         )
            
#     #         response.raise_for_status()
#     #         result = response.json()
            
#     #         # Extract the analysis from the response
#     #         if "choices" in result and len(result["choices"]) > 0:
#     #             analysis = result["choices"][0]["message"]["content"]
                
#     #             logger.info(f"✅ Successfully analyzed image: {image_path}")
                
#     #             return {
#     #                 "success": True,
#     #                 "analysis": analysis,
#     #                 "model": self.model,
#     #                 "image_path": image_path,
#     #                 "prompt_used": prompt,
#     #                 "usage": result.get("usage", {})
#     #             }
#     #         else:
#     #             logger.error("❌ No analysis returned from DeepSeek Vision")
#     #             return {
#     #                 "success": False,
#     #                 "error": "No analysis returned from API",
#     #                 "analysis": None
#     #             }
                
#     #     except requests.exceptions.RequestException as e:
#     #         logger.error(f"❌ Request error in DeepSeek Vision API: {str(e)}")
#     #         return {
#     #             "success": False,
#     #             "error": f"API request failed: {str(e)}",
#     #             "analysis": None
#     #         }
#     #     except Exception as e:
#     #         logger.error(f"❌ Error analyzing image with DeepSeek Vision: {str(e)}")
#     #         return {
#     #             "success": False,
#     #             "error": str(e),
#     #             "analysis": None
#     #         }

#     def __init__(self):
#         model_path = "deepseek-ai/deepseek-vl-7b-base"
#         vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
#         tokenizer = vl_chat_processor.tokenizer

#         vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
#         vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

#         conversation = [
#             {
#                 "role": "User",
#                 "content": "<image_placeholder>Describe each stage of this image.",
#                 "images": ["./downloads/whatsapp_image_94713966820_20250901_230730_17511838.jpg"]
#             },
#             {
#                 "role": "Assistant",
#                 "content": ""
#             }
#         ]

#         # load images and prepare for inputs
#         pil_images = load_pil_images(conversation)
#         prepare_inputs = vl_chat_processor(
#             conversations=conversation,
#             images=pil_images,
#             force_batchify=True
#         ).to(vl_gpt.device)

#         # run image encoder to get the image embeddings
#         inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

#         # run the model to get the response
#         outputs = vl_gpt.language_model.generate(
#             inputs_embeds=inputs_embeds,
#             attention_mask=prepare_inputs.attention_mask,
#             pad_token_id=tokenizer.eos_token_id,
#             bos_token_id=tokenizer.bos_token_id,
#             eos_token_id=tokenizer.eos_token_id,
#             max_new_tokens=512,
#             do_sample=False,
#             use_cache=True
#         )

#         answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
#         logger.info(f"✅ Successfully analyzed image with DeepSeek Vision: {answer}")
#         print(f"{prepare_inputs['sft_format'][0]}", answer)
    
#     def analyze_image_for_business(self, image_path: str, business_context: str = "", language: str = "en") -> Dict[str, Any]:
#         """
#         Analyze an image with business context (e.g., product inquiry, complaint, etc.)
        
#         Args:
#             image_path: Path to the image file
#             business_context: Context about the business/product
#             language: Language for the response
            
#         Returns:
#             Dictionary containing business-focused analysis
#         """
#         if language == "ar":
#             business_prompt = f"""Analyze this image from a business perspective. {business_context}

#             Please identify:
#             1. Type of product or service shown
#             2. Condition of the product (new, used, damaged, etc.)
#             3. Any visible problems or defects
#             4. Image quality and clarity
#             5. Important recommendations or observations
#             6. Any important text or numbers in the image
#             7. Is this a payment confirmation?

#             Be precise and helpful in your analysis."""
        
#         return self.analyze_image(image_path, business_prompt, language)
    
#     def extract_text_from_image(self, image_path: str, language: str = "en") -> Dict[str, Any]:
#         """
#         Extract text content from an image using DeepSeek Vision
        
#         Args:
#             image_path: Path to the image file
#             language: Language for instructions
            
#         Returns:
#             Dictionary containing extracted text
#         """
#         if language == "ar":
#             text_prompt = """Extract all text content from this image.
# Write the text exactly as it appears, maintaining formatting and order.
# If text is in multiple languages, write each text in its original language."""
        
#         return self.analyze_image(image_path, text_prompt, language)
    
#     def is_configured(self) -> bool:
#         """Check if the service is properly configured"""
#         return bool(self.api_key)
    
#     def get_supported_formats(self) -> List[str]:
#         """Get list of supported image formats"""
#         return [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]

# # Global instance
# deepseek_vision_service = DeepSeekVisionService()