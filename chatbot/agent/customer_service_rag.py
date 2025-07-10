"""
Enhanced RAG Agent specifically designed for customer service using Bitext dataset
"""

from typing import Dict, List, Any
from utils.logger import get_logger
from vector_store.vector_store import vector_store
import json

logger = get_logger(__name__)

class CustomerServiceRAGAgent:
    """
    Specialized RAG agent for customer service using the Bitext dataset
    """
    
    def __init__(self):
        self.vector_store = vector_store
        self.intent_mapping = self._load_intent_mapping()
        
    def _load_intent_mapping(self) -> Dict[str, str]:
        """Load Bitext intent categories mapping"""
        return {
            # ACCOUNT category
            "create_account": "ACCOUNT",
            "delete_account": "ACCOUNT", 
            "edit_account": "ACCOUNT",
            "switch_account": "ACCOUNT",
            
            # CANCELLATION_FEE category
            "check_cancellation_fee": "CANCELLATION_FEE",
            
            # DELIVERY category
            "delivery_options": "DELIVERY",
            
            # FEEDBACK category
            "complaint": "FEEDBACK",
            "review": "FEEDBACK",
            
            # INVOICE category
            "check_invoice": "INVOICE",
            "get_invoice": "INVOICE",
            
            # NEWSLETTER category
            "newsletter_subscription": "NEWSLETTER",
            
            # ORDER category
            "cancel_order": "ORDER",
            "change_order": "ORDER",
            "place_order": "ORDER",
            
            # PAYMENT category
            "check_payment_methods": "PAYMENT",
            "payment_issue": "PAYMENT",
            
            # REFUND category
            "check_refund_policy": "REFUND",
            "track_refund": "REFUND",
            
            # SHIPPING_ADDRESS category
            "change_shipping_address": "SHIPPING_ADDRESS",
            "set_up_shipping_address": "SHIPPING_ADDRESS"
        }
    
    def analyze_query_context(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze user query to extract context and suggest best RAG strategy
        """
        query_lower = user_query.lower()
        
        # Detect query type
        query_type = "general"
        confidence = 0.5
        
        # Account-related keywords
        if any(word in query_lower for word in ["account", "profile", "login", "register", "sign up"]):
            query_type = "account"
            confidence = 0.8
            
        # Order-related keywords  
        elif any(word in query_lower for word in ["order", "purchase", "buy", "cart", "checkout"]):
            query_type = "order"
            confidence = 0.8
            
        # Payment-related keywords
        elif any(word in query_lower for word in ["payment", "pay", "credit card", "billing", "charge"]):
            query_type = "payment" 
            confidence = 0.8
            
        # Refund-related keywords
        elif any(word in query_lower for word in ["refund", "return", "money back", "cancel"]):
            query_type = "refund"
            confidence = 0.8
            
        # Delivery/shipping keywords
        elif any(word in query_lower for word in ["delivery", "shipping", "ship", "address", "location"]):
            query_type = "delivery"
            confidence = 0.8
            
        # Support/complaint keywords
        elif any(word in query_lower for word in ["complaint", "problem", "issue", "help", "support"]):
            query_type = "support"
            confidence = 0.7
            
        return {
            "query_type": query_type,
            "confidence": confidence,
            "suggested_category": query_type.upper(),
            "keywords_found": [word for word in query_lower.split() if len(word) > 2]
        }
    
    def get_relevant_examples(self, user_query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant examples from Bitext dataset based on user query
        """
        logger.info(f"[RAG Agent] Getting relevant examples for query: {user_query[:50]}...")
        
        try:
            # Get context analysis
            context = self.analyze_query_context(user_query)
            logger.debug(f"[RAG Agent] Query context: {context}")
            
            # Primary search: semantic similarity
            results = self.vector_store.similarity_search(user_query, k=k)
            
            # If no good results, try category-based search
            if len(results) < k//2 and context["confidence"] > 0.7:
                category_results = self.vector_store.search_by_category(
                    context["suggested_category"], k=k//2
                )
                results.extend(category_results)
            
            # Format results
            formatted_results = []
            for result in results:
                if hasattr(result, 'metadata') and result.metadata.get('source') == 'bitext_customer_service':
                    formatted_results.append({
                        "instruction": result.page_content,
                        "response": result.metadata.get('response', ''),
                        "intent": result.metadata.get('intent', ''),
                        "category": result.metadata.get('category', ''),
                        "flags": result.metadata.get('flags', ''),
                        "relevance_score": context["confidence"]
                    })
            
            logger.info(f"[RAG Agent] Found {len(formatted_results)} relevant examples")
            return formatted_results[:k]
            
        except Exception as e:
            logger.error(f"[RAG Agent] Error getting relevant examples: {str(e)}")
            return []
    
    def get_intent_specific_examples(self, intent: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Get examples for a specific intent from the Bitext dataset
        """
        try:
            results = self.vector_store.search_by_intent(intent, k=k)
            
            formatted_results = []
            for result in results:
                if hasattr(result, 'metadata'):
                    formatted_results.append({
                        "instruction": result.page_content,
                        "response": result.metadata.get('response', ''),
                        "intent": result.metadata.get('intent', ''),
                        "category": result.metadata.get('category', ''),
                        "flags": result.metadata.get('flags', '')
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[RAG Agent] Error getting intent examples: {str(e)}")
            return []
    
    def generate_contextual_response(self, user_query: str, intent: str = None) -> Dict[str, Any]:
        """
        Generate a contextual response using RAG with Bitext examples
        """
        logger.info(f"[RAG Agent] Generating contextual response for: {user_query[:50]}...")
        
        try:
            # Get relevant examples
            examples = self.get_relevant_examples(user_query)
            
            # If intent is provided, get intent-specific examples too
            if intent:
                intent_examples = self.get_intent_specific_examples(intent)
                examples.extend(intent_examples)
            
            # Remove duplicates
            seen_instructions = set()
            unique_examples = []
            for example in examples:
                if example["instruction"] not in seen_instructions:
                    unique_examples.append(example)
                    seen_instructions.add(example["instruction"])
            
            # Analyze patterns in examples
            response_patterns = self._analyze_response_patterns(unique_examples)
            
            result = {
                "examples": unique_examples[:5],  # Top 5 examples
                "response_patterns": response_patterns,
                "suggested_approach": self._suggest_response_approach(unique_examples),
                "context_analysis": self.analyze_query_context(user_query)
            }
            
            logger.info(f"[RAG Agent] Generated contextual response with {len(unique_examples)} examples")
            return result
            
        except Exception as e:
            logger.error(f"[RAG Agent] Error generating contextual response: {str(e)}")
            return {
                "examples": [],
                "response_patterns": {},
                "suggested_approach": "general_assistance",
                "context_analysis": {}
            }
    
    def _analyze_response_patterns(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in the response examples
        """
        if not examples:
            return {}
        
        # Common response starters
        starters = []
        # Common intents
        intents = []
        # Common categories
        categories = []
        # Language flags
        flags = []
        
        for example in examples:
            response = example.get("response", "")
            if response:
                # Get first few words as starter
                words = response.split()
                if len(words) >= 2:
                    starters.append(" ".join(words[:3]))
            
            if example.get("intent"):
                intents.append(example["intent"])
            if example.get("category"):
                categories.append(example["category"])
            if example.get("flags"):
                flags.append(example["flags"])
        
        return {
            "common_starters": list(set(starters))[:3],
            "primary_intents": list(set(intents)),
            "primary_categories": list(set(categories)),
            "language_styles": list(set(flags))[:3]
        }
    
    def _suggest_response_approach(self, examples: List[Dict[str, Any]]) -> str:
        """
        Suggest the best response approach based on examples
        """
        if not examples:
            return "general_assistance"
        
        # Count categories to determine approach
        categories = [ex.get("category", "") for ex in examples]
        
        if "ORDER" in categories:
            return "order_assistance"
        elif "ACCOUNT" in categories:
            return "account_assistance"
        elif "PAYMENT" in categories:
            return "payment_assistance"
        elif "REFUND" in categories:
            return "refund_assistance"
        elif "DELIVERY" in categories:
            return "delivery_assistance"
        else:
            return "general_customer_service"
    
    def get_available_intents(self) -> List[str]:
        """Get all available intents from the dataset"""
        return self.vector_store.get_available_intents()
    
    def get_available_categories(self) -> List[str]:
        """Get all available categories from the dataset"""  
        return self.vector_store.get_available_categories()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded Bitext dataset"""
        return self.vector_store.get_dataset_info()

# Create global instance
customer_service_rag = CustomerServiceRAGAgent()
