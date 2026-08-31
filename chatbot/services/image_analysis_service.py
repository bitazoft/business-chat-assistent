"""
Vision service - lets the assistant actually SEE customer images.

Uses the same OpenAI-compatible endpoint as the chat LLM (OpenRouter by default),
so no extra provider or API key is needed. The model is set by VISION_MODEL and
must be vision-capable (e.g. openai/gpt-4o-mini).

Three jobs:
  1. classify_image()          - what kind of image is this? (receipt / product / other)
  2. extract_receipt_details() - read amount, reference, bank, date off a payment receipt
  3. describe_product_image()  - turn a product photo into search text
"""
import base64
import json
import os
import mimetypes
from typing import Dict, Any, Optional

import requests
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB - keeps request payloads sane


class VisionService:
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.api_base = os.getenv("API_BASE", "https://openrouter.ai/api/v1").rstrip("/")
        # Falls back to the chat model, which is vision-capable on the default setup.
        self.model = os.getenv("VISION_MODEL") or os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
        self.timeout = int(os.getenv("VISION_TIMEOUT", "60"))

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_supported_formats(self) -> list:
        return sorted(SUPPORTED_FORMATS)

    # ------------------------------------------------------------------ helpers

    def _encode_image(self, image_path: str) -> Dict[str, Any]:
        """Read an image off disk and return a base64 data URI."""
        if not os.path.exists(image_path):
            return {"success": False, "error": f"File not found: {image_path}"}

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            return {"success": False, "error": f"Unsupported image format '{ext}'"}

        size = os.path.getsize(image_path)
        if size > MAX_IMAGE_BYTES:
            return {"success": False, "error": f"Image too large ({size} bytes, max {MAX_IMAGE_BYTES})"}

        mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        return {"success": True, "data_uri": f"data:{mime_type};base64,{encoded}"}

    def _ask_vision(self, image_path: str, prompt: str, max_tokens: int = 700) -> Dict[str, Any]:
        """Send one image + prompt to the vision model, return raw text."""
        if not self.is_configured():
            return {"success": False, "error": "Vision service not configured - API_KEY is missing"}

        encoded = self._encode_image(image_path)
        if not encoded["success"]:
            return encoded

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # deterministic - we want facts off the image, not creativity
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": encoded["data_uri"]}},
                ],
            }],
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                logger.error(f"Vision API error {response.status_code}: {response.text[:300]}")
                return {"success": False, "error": f"Vision API returned {response.status_code}"}

            content = response.json()["choices"][0]["message"]["content"]
            return {"success": True, "analysis": content}

        except requests.Timeout:
            return {"success": False, "error": f"Vision request timed out after {self.timeout}s"}
        except Exception as e:
            logger.error(f"Vision request failed: {str(e)}")
            return {"success": False, "error": str(e)}

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Vision models often wrap JSON in prose or ``` fences - dig it out."""
        if not text:
            return None
        cleaned = text.strip()
        if "```" in cleaned:
            # take the content of the first fenced block
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1]
                if cleaned.lstrip().lower().startswith("json"):
                    cleaned = cleaned.lstrip()[4:]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            logger.warning(f"Could not parse vision JSON: {cleaned[:200]}")
            return None

    # ------------------------------------------------------------------- public

    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        Work out what the customer sent us.

        Returns image_type: 'payment_receipt' | 'product_photo' | 'other'
        """
        prompt = (
            "Look at this image sent by a customer to an online shop and classify it.\n"
            "Reply with ONLY a JSON object, no other text:\n"
            "{\n"
            '  "image_type": "payment_receipt" | "product_photo" | "other",\n'
            '  "confidence": 0.0-1.0,\n'
            '  "summary": "one short sentence describing what is in the image"\n'
            "}\n\n"
            "Guidance:\n"
            "- 'payment_receipt': a bank transfer slip, ATM/deposit slip, online banking confirmation, "
            "mobile wallet transfer screenshot, or any proof of payment.\n"
            "- 'product_photo': a photo or screenshot of a physical item the customer might want to buy "
            "or ask about.\n"
            "- 'other': anything else (selfies, memes, documents, screenshots of chats, blank images)."
        )

        result = self._ask_vision(image_path, prompt, max_tokens=200)
        if not result["success"]:
            return {"success": False, "error": result["error"], "image_type": "unknown"}

        parsed = self._parse_json(result["analysis"])
        if not parsed:
            return {
                "success": False,
                "error": "Could not understand the vision model response",
                "image_type": "unknown",
                "raw": result["analysis"],
            }

        return {
            "success": True,
            "image_type": parsed.get("image_type", "other"),
            "confidence": parsed.get("confidence", 0.0),
            "summary": parsed.get("summary", ""),
        }

    def extract_receipt_details(self, image_path: str) -> Dict[str, Any]:
        """
        Read a payment receipt. Returns whether it really is a receipt plus the
        details needed to check it against an order.
        """
        prompt = (
            "This image should be a proof of payment (bank transfer slip, deposit slip, "
            "online banking confirmation, or mobile wallet transfer).\n"
            "Read it carefully and reply with ONLY a JSON object, no other text:\n"
            "{\n"
            '  "is_receipt": true | false,\n'
            '  "amount": number or null,\n'
            '  "currency": "LKR" | "USD" | other code | null,\n'
            '  "reference": "transaction/reference number" or null,\n'
            '  "bank": "bank or service name" or null,\n'
            '  "date": "YYYY-MM-DD" or null,\n'
            '  "beneficiary": "who received the money" or null,\n'
            '  "confidence": 0.0-1.0,\n'
            '  "issues": ["anything that looks wrong, unreadable, edited, or suspicious"]\n'
            "}\n\n"
            "Rules:\n"
            "- Set is_receipt to false if this is not a proof of payment at all.\n"
            "- Use null for any field you genuinely cannot read. Never guess a number.\n"
            "- 'amount' must be the amount transferred, as a plain number without commas or currency symbols.\n"
            "- Note in 'issues' if the image is blurry, cropped, obviously edited, or the amount is unclear."
        )

        result = self._ask_vision(image_path, prompt, max_tokens=500)
        if not result["success"]:
            return {"success": False, "error": result["error"]}

        parsed = self._parse_json(result["analysis"])
        if not parsed:
            return {
                "success": False,
                "error": "Could not understand the vision model response",
                "raw": result["analysis"],
            }

        # Normalise amount - models sometimes return "1,500.00" despite instructions
        amount = parsed.get("amount")
        if isinstance(amount, str):
            try:
                amount = float(amount.replace(",", "").strip())
            except ValueError:
                amount = None

        return {
            "success": True,
            "is_receipt": bool(parsed.get("is_receipt", False)),
            "amount": amount,
            "currency": parsed.get("currency"),
            "reference": parsed.get("reference"),
            "bank": parsed.get("bank"),
            "date": parsed.get("date"),
            "beneficiary": parsed.get("beneficiary"),
            "confidence": parsed.get("confidence", 0.0),
            "issues": parsed.get("issues") or [],
        }

    def describe_product_image(self, image_path: str) -> Dict[str, Any]:
        """
        Turn a product photo into search text we can match against the catalogue.
        """
        prompt = (
            "A customer sent this photo asking about a product.\n"
            "Describe the item so it can be matched against a shop's product catalogue.\n"
            "Reply with ONLY a JSON object, no other text:\n"
            "{\n"
            '  "item_type": "what the item is, e.g. t-shirt, running shoe, ceramic mug",\n'
            '  "colors": ["main colours"],\n'
            '  "material": "material if visible" or null,\n'
            '  "brand": "brand if clearly visible" or null,\n'
            '  "keywords": ["5-10 words a shop might use to name or describe this item"],\n'
            '  "description": "one natural sentence describing the item"\n'
            "}\n\n"
            "Describe only what you can actually see. Use null or empty lists when unsure."
        )

        result = self._ask_vision(image_path, prompt, max_tokens=400)
        if not result["success"]:
            return {"success": False, "error": result["error"]}

        parsed = self._parse_json(result["analysis"])
        if not parsed:
            return {
                "success": False,
                "error": "Could not understand the vision model response",
                "raw": result["analysis"],
            }

        keywords = parsed.get("keywords") or []
        colors = parsed.get("colors") or []

        # One flat string for keyword matching against product name/description
        search_terms = " ".join(
            str(t) for t in ([parsed.get("item_type")] + colors + keywords + [parsed.get("brand")]) if t
        )

        return {
            "success": True,
            "item_type": parsed.get("item_type"),
            "colors": colors,
            "material": parsed.get("material"),
            "brand": parsed.get("brand"),
            "keywords": keywords,
            "description": parsed.get("description", ""),
            "search_terms": search_terms,
        }

    def analyze_image(self, image_path: str, prompt: str, language: str = "en") -> Dict[str, Any]:
        """Free-form image question - used by the manual /analyze-image endpoint."""
        if language and language != "en":
            prompt = f"{prompt}\n\nRespond in language code: {language}."
        return self._ask_vision(image_path, prompt)

    def extract_text_from_image(self, image_path: str, language: str = "en") -> Dict[str, Any]:
        """Plain OCR - used by the manual /extract-text endpoint."""
        return self._ask_vision(
            image_path,
            "Extract all readable text from this image exactly as it appears. "
            "Return only the text, no commentary. If there is no text, return an empty response.",
        )


# Singleton used across the app
vision_service = VisionService()

# Backwards-compatible alias - whatsapp_routes.py referred to this name
deepseek_vision_service = vision_service
