"""
Language Detection Agent
A specialized agent for identifying Sinhala, Singlish, and English languages
"""

from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
import re
import os
from functools import lru_cache
from utils.logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Configure LLM for language detection
language_detection_llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
    temperature=0.0,  # Very low temperature for consistent classification
    max_tokens=50,    # Very short responses for speed
    timeout=60,       # Quick timeout
    max_retries=2
)

class LanguageDetectionResult(BaseModel):
    """Result of language detection"""
    language: str = Field(description="Detected language: 'sinhala', 'singlish', or 'english'")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    indicators: List[str] = Field(description="List of indicators that led to this classification")

class LanguageAgent:
    """Specialized agent for language detection"""
    
    def __init__(self):
        self.llm = language_detection_llm
        self.parser = StrOutputParser()
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
        
        # Language patterns and indicators
        self.sinhala_unicode_pattern = re.compile(r'[\u0D80-\u0DFF]')
        self.sinhala_words = self._load_sinhala_vocabulary()
        self.singlish_phrases = self._load_singlish_phrases()
        self.english_indicators = self._load_english_indicators()
    
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create prompt for language detection"""
        return ChatPromptTemplate.from_template("""
You are a language classification expert specializing in Sri Lankan languages.

Analyze the following text and classify it as one of these languages:
- "sinhala": Pure Sinhala language (either in Sinhala script or romanized)
- "singlish": Mixed Sinhala-English (Sri Lankan colloquial style)
- "english": Pure English

Text to analyze: "{text}"

Guidelines:
1. If text contains Sinhala Unicode characters (අ, ආ, ඇ, etc.), classify as "sinhala"
2. If text contains multiple Sinhala words in Latin script mixed with English, classify as "singlish"
3. If text contains Sri Lankan expressions like "machang", "aiyo", "aney", classify as "singlish"
4. If text is purely English without Sri Lankan terms, classify as "english"

Respond with ONLY the classification: sinhala, singlish, or english
""")
    
    def _load_sinhala_vocabulary(self) -> List[str]:
        """Load comprehensive Sinhala vocabulary in Latin script"""
        return [
            # Basic words
            'mama', 'oba', 'api', 'eyala', 'eya', 'eka', 'meka', 'dan', 'heta',
            'ige', 'wage', 'nisa', 'hinda', 'ekka', 'nattam', 'oyata', 'umbata',
            'apita', 'ekata', 'mokak', 'kohomada', 'mokakda', 'kiyala', 'denna',
            'ganna', 'yanna', 'enna', 'awa', 'naha', 'hari', 'hodai', 'wadai',
            
            # Family terms
            'amma', 'thatha', 'aiya', 'akka', 'malli', 'nangi', 'seeya', 'achchi',
            'putha', 'duwa', 'redda', 'lamaya', 'miniha', 'gahaniya', 'kella',
            
            # Common verbs
            'karanna', 'kiyanna', 'balanna', 'ahanna', 'bonawa', 'nathuwa',
            'kiyanawa', 'karala', 'giyapu', 'enawa', 'yanawa', 'dennawa',
            'gannawa', 'pennawa', 'ahanawa', 'kanawa', 'boyanawa',
            
            # Adjectives and descriptors
            'honda', 'naraka', 'loku', 'podi', 'alut', 'parana', 'issara',
            'godak', 'tikak', 'pana', 'rupi', 'badu', 'potta',
            
            # Questions and expressions
            'kiyada', 'tiyenawa', 'thiyen', 'puluwan', 'bari', 'giya', 'aawa',
            'kada', 'wela', 'passe', 'issara', 'adalath', 'sampurnayenma',
            
            # Shopping/business related
            'ganna', 'kirana', 'gannam', 'aragenna', 'parakkuwada', 'order',
            'delivery', 'anawum', 'wisthara', 'profile', 'update', 'change'
        ]
    
    def _load_singlish_phrases(self) -> List[str]:
        """Load Singlish expressions and mixed phrases"""
        return [
            # Classic Singlish
            'aiyo', 'aney', 'machang', 'machan', 'pako', 'puthanoo', 'sudda',
            'hora', 'bola', 'pissek', 'adurei', 'chandiya', 'bugger', 'fellows',
            
            # Mixed expressions
            'no no', 'ban kara', 'hari awa', 'wada nehe', 'giyapu wela',
            'ekkenek', 'menawada', 'therenne', 'dannawa', 'kiwwe',
            
            # Sri Lankan English
            'put lights', 'take air', 'fall down', 'get down', 'come soon',
            'go and come', 'put on the fan', 'off the lights', 'on the TV',
            
            # Common Singlish patterns
            'mata thiyen', 'oba ge', 'mama ge', 'api ge', 'order eka',
            'price eka', 'delivery kada', 'awa da', 'thiyen da', 'karanna ona',
            'ganna ona', 'buy karanna', 'order karanna', 'track karanna',
            'update karanna', 'change karanna', 'kohomada', 'kiyada wage'
        ]
    
    def _load_english_indicators(self) -> List[str]:
        """Load pure English indicators"""
        return [
            'the', 'and', 'or', 'but', 'with', 'without', 'through', 'about',
            'would', 'could', 'should', 'might', 'will', 'shall', 'can',
            'please', 'thank', 'welcome', 'sorry', 'excuse', 'help',
            'information', 'available', 'service', 'product', 'order',
            'delivery', 'tracking', 'account', 'profile', 'update'
        ]
    
    def _rule_based_detection(self, text: str) -> Tuple[str, float, List[str]]:
        """Fast rule-based language detection"""
        text_lower = text.lower().strip()
        indicators = []
        
        # Check for Sinhala Unicode characters
        if self.sinhala_unicode_pattern.search(text):
            return 'sinhala', 0.95, ['sinhala_unicode_detected']
        
        # Count word matches
        sinhala_matches = [word for word in self.sinhala_words if word in text_lower]
        singlish_matches = [phrase for phrase in self.singlish_phrases if phrase in text_lower]
        english_matches = [word for word in self.english_indicators if word in text_lower]
        
        # Calculate scores
        total_words = len(text_lower.split())
        sinhala_score = len(sinhala_matches) / max(total_words, 1)
        singlish_score = len(singlish_matches) / max(total_words, 1)
        english_score = len(english_matches) / max(total_words, 1)
        
        # Decision logic
        if sinhala_score >= 0.3:
            if singlish_score > 0 or any(eng in text_lower for eng in ['i', 'you', 'can', 'please']):
                indicators.extend(['mixed_sinhala_english', f'sinhala_words: {sinhala_matches[:3]}'])
                return 'singlish', min(0.9, 0.6 + sinhala_score), indicators
            else:
                indicators.extend(['pure_sinhala', f'sinhala_words: {sinhala_matches[:3]}'])
                return 'sinhala', min(0.9, 0.7 + sinhala_score), indicators
        
        elif singlish_score > 0:
            indicators.extend(['singlish_expressions', f'singlish_phrases: {singlish_matches[:3]}'])
            return 'singlish', min(0.85, 0.6 + singlish_score), indicators
        
        elif sinhala_score > 0 and english_score > 0:
            indicators.extend(['mixed_content', f'sinhala: {sinhala_matches[:2]}, english: {english_matches[:2]}'])
            return 'singlish', 0.7, indicators
        
        else:
            indicators.extend(['english_default', f'english_words: {english_matches[:3]}'])
            return 'english', min(0.8, 0.5 + english_score), indicators
    
    @lru_cache(maxsize=500)
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        Detect language using both rule-based and LLM approaches
        """
        try:
            # First try rule-based detection (fast)
            rule_language, rule_confidence, rule_indicators = self._rule_based_detection(text)
            
            # If confidence is high enough, return rule-based result
            if rule_confidence >= 0.85:
                logger.info(f"[LanguageAgent] Rule-based detection: {rule_language} (confidence: {rule_confidence:.2f})")
                return LanguageDetectionResult(
                    language=rule_language,
                    confidence=rule_confidence,
                    indicators=rule_indicators
                )
            
            # For lower confidence, use LLM for verification
            try:
                llm_result = self.chain.invoke({"text": text}).strip().lower()
                
                # Validate LLM result
                if llm_result in ['sinhala', 'singlish', 'english']:
                    # Combine rule-based and LLM results
                    if llm_result == rule_language:
                        final_confidence = min(0.95, rule_confidence + 0.2)
                    else:
                        final_confidence = 0.6  # Lower confidence when methods disagree
                        rule_indicators.append(f'llm_suggests: {llm_result}')
                    
                    logger.info(f"[LanguageAgent] Combined detection: {rule_language} (rule: {rule_confidence:.2f}, llm: {llm_result})")
                    return LanguageDetectionResult(
                        language=rule_language if final_confidence > 0.6 else llm_result,
                        confidence=final_confidence,
                        indicators=rule_indicators
                    )
                else:
                    # Invalid LLM result, fallback to rule-based
                    logger.warning(f"[LanguageAgent] Invalid LLM result: {llm_result}, using rule-based")
                    
            except Exception as e:
                logger.warning(f"[LanguageAgent] LLM detection failed: {str(e)}, using rule-based")
            
            # Fallback to rule-based result
            return LanguageDetectionResult(
                language=rule_language,
                confidence=rule_confidence,
                indicators=rule_indicators
            )
            
        except Exception as e:
            logger.error(f"[LanguageAgent] Detection failed: {str(e)}")
            return LanguageDetectionResult(
                language='english',
                confidence=0.3,
                indicators=['error_fallback']
            )
    
    def detect_language_simple(self, text: str) -> str:
        """Simple interface that returns just the language string"""
        result = self.detect_language(text)
        return result.language
    
    def batch_detect(self, texts: List[str]) -> List[LanguageDetectionResult]:
        """Detect language for multiple texts"""
        return [self.detect_language(text) for text in texts]

# Global language agent instance
_language_agent = None

def get_language_agent() -> LanguageAgent:
    """Get singleton language agent instance"""
    global _language_agent
    if _language_agent is None:
        _language_agent = LanguageAgent()
    return _language_agent

# Convenience functions
def detect_language(text: str) -> str:
    """Quick language detection - returns language string"""
    agent = get_language_agent()
    return agent.detect_language_simple(text)

def detect_language_detailed(text: str) -> LanguageDetectionResult:
    """Detailed language detection - returns full result"""
    agent = get_language_agent()
    return agent.detect_language(text)
