from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool, StructuredTool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from db.database import SessionLocal
import pandas as pd
from utils.logger import get_logger
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Get logger for this module
logger = get_logger(__name__)

# from models.schemas import Product, Order, ChatLog, OrderItem, User
from repositories.tools import (
    get_product_info,
    track_order,
    place_order,
    log_query,
    get_user_info,
    update_user_info,
    check_user_exists,
    save_user,
    create_tmp_user_id,
    get_all_products
)
from vector_store.vector_store import vector_store
import os
import numpy as np
import requests
from datetime import datetime
from functools import partial

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Make functions available for import
__all__ = [
    'create_multi_agent_system',
    'log_query',
    'get_product_info',
    'track_order',
    'place_order',
    'get_user_info',
    'update_user_info',
    'check_user_exists',
    'save_user',
    'create_tmp_user_id'
]
# LangChain Agent Setup
llm = ChatDeepSeek(
    model="deepseek-chat",  # Replace with actual DeepSeek model name
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE
)

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", f"""You are a business assistant for seller ID: {seller_id}. Use tools to fetch product info, track orders, or place orders. The seller_id and user_id are automatically provided to tools that need them. 

    #     USER MANAGEMENT WORKFLOW:
    #     - For general inquiries (product info, order tracking, questions): No need to collect user details
    #     - ONLY when user wants to PLACE AN ORDER:
    #       1. Check if user exists using check_user_exists
    #       2. If user doesn't exist, ask for their details (name, email, address, phone number) and create them using save_user
    #       3. If user exists, get their info using get_user_info and show it to confirm details are correct
    #       4. If user wants to update any information, use update_user_info with only the fields they want to change
    #       5. Only proceed with placing order after confirming user details

    #     GENERAL INSTRUCTIONS:
    #     - Log all queries using log_query
    #     - For product questions, use get_product_info (no user details needed)
    #     - For order tracking, use track_order (no user details needed)
    #     - For placing orders, use place_order (user details required first)
    #     - Always be helpful and only ask for information when necessary"""),
    #     MessagesPlaceholder(variable_name="chat_history"),
    #     ("human", "{input}"),
    #     MessagesPlaceholder(variable_name="agent_scratchpad")
    # ])

# # Load vector store
# df = pd.read_csv("your_dataset.csv")
# embeddings = DeepSeekEmbeddings(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
# vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# unique_intents = df["intent"].unique().tolist()

# Helper function for RAG
def get_rag_examples(user_input, seller_id, k=3):
    logger.debug(f"[RAG Helper] Getting RAG examples for input: {user_input}, seller_id: {seller_id}")
    
    try:
        results = vector_store.similarity_search(user_input, k=k)
        examples = "\n".join([
            f"Instruction: {result.page_content}, Intent: {result.metadata['intent']}, Response: {result.metadata['response']}"
            for result in results if result.metadata.get('category') == seller_id or not result.metadata.get('category')
        ])
        
        logger.debug(f"[RAG Helper] Found {len(results)} results, filtered examples generated")
        return examples
    except Exception as e:
        logger.error(f"[RAG Helper] Error getting RAG examples: {str(e)}")
        return ""

# Helper function to safely parse JSON responses from LLM
def parse_llm_json_response(response_text, default_value=None):
    """
    Safely parse JSON response from LLM, handling markdown code blocks and other formatting issues.
    
    Args:
        response_text (str): The raw response from LLM
        default_value: Value to return if parsing fails (default: None)
    
    Returns:
        dict: Parsed JSON data or default_value if parsing fails
    """
    logger.debug(f"[JSON Parser] Parsing LLM response: {response_text[:100]}...")
    
    try:
        # Clean the response text
        cleaned_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        elif cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
            
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
            
        cleaned_text = cleaned_text.strip()
        
        # Parse JSON
        parsed_data = json.loads(cleaned_text)
        logger.debug(f"[JSON Parser] Successfully parsed JSON: {parsed_data}")
        return parsed_data
        
    except (json.JSONDecodeError, AttributeError) as e:
        logger.error(f"[JSON Parser] Failed to parse JSON: {str(e)}")
        logger.debug(f"[JSON Parser] Raw response: {response_text}")
        logger.debug(f"[JSON Parser] Cleaned text: {cleaned_text}")
        return default_value if default_value is not None else {}

unique_intents = ["product_info", "order_tracking", "place_order", "user_management", "general_inquiry"]  #need to implement rag

# Agent Definitions
def create_intent_agent(seller_id):
    logger.info(f"[Intent Agent] Creating intent agent for seller_id: {seller_id}")
    
    # First chain: Extract key entities and context
    entity_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Extract key entities and context from the user input for seller ID: """ + seller_id + """.
            Look for: product names, order IDs, user information, quantities, etc.
            
            Examples: {examples}
            
            Respond with JSON: "entities": {{"products": [], "user_info": [], "other": []}}, "context": "<summary>"
         
        """),
        ("human", "{input}")
    ])
    
    # Second chain: Classify intent based on entities and context
    intent_classification_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Based on the extracted entities and context: {entity_analysis}
            
            Classify the intent from these available intents: {intents}
            
            Consider:
            1. The extracted entities
            2. The context summary
            3. The user's likely goal
            
            Respond with JSON: {{"intent": "<intent>", "confidence": 0.0-1.0, "reasoning": "<explanation>"}}
        """),
        ("human", "Original input: {input}")
    ])
    
    # Chain them together
    def intent_chain(input_data):
        logger.info(f"[Intent Agent] Processing input for seller_id: {seller_id}")
        logger.debug(f"[Intent Agent] Input data: {input_data}")
        
        try:
            # Step 1: Extract entities
            logger.info("[Intent Agent] Step 1: Extracting entities")
            entity_result = (
                {"input": RunnablePassthrough(), "examples": lambda x: get_rag_examples(x, seller_id)}
                | entity_extraction_prompt
                | llm
                | StrOutputParser()
            ).invoke(input_data)
            logger.debug(f"[Intent Agent] Entity extraction result: {entity_result}")
            # Step 2: Classify intent
            logger.info("[Intent Agent] Step 2: Classifying intent")
            intent_result = (
                {
                    "input": RunnablePassthrough(), 
                    "entity_analysis": lambda _: entity_result,
                    "intents": lambda _: ", ".join(unique_intents)
                }
                | intent_classification_prompt
                | llm
                | StrOutputParser()
            ).invoke(input_data)
            logger.info(f"[Intent Agent] Intent classification completed: {intent_result}")
            
            return intent_result
        except Exception as e:
            logger.error(f"[Intent Agent] Error in intent processing: {str(e)}")
            raise
    
    return intent_chain

def create_function_agent(seller_id: str, user_id: str):
    logger.info(f"[Function Agent] Creating unified function agent for seller_id: {seller_id}, user_id: {user_id}")

    # Define Pydantic schemas for tools with parameters
    class GetProductInfoInput(BaseModel):
        product_name: str = Field(description="Name of the product to retrieve details for")

    class TrackOrderInput(BaseModel):
        order_id: str = Field(description="Order ID to track the status of")

    class PlaceOrderInput(BaseModel):
        items: List[dict] = Field(description="List of items, each with product_id (string) and quantity (integer)")

    class SaveUserInput(BaseModel):
        name: str = Field(description="User's full name")
        email: str = Field(description="User's email address")
        address: str = Field(description="User's physical address")
        number: str = Field(description="User's phone number")

    class UpdateUserInfoInput(BaseModel):
        name: str | None = Field(description="User's full name", default="")
        email: str | None = Field(description="User's email address", default="")
        address: str | None = Field(description="User's physical address", default="")
        number: str | None = Field(description="User's phone number", default="")

    class EmptyInput(BaseModel):
        pass  # For tools with no parameters

    # Define wrapper functions to embed seller_id and user_id
    def get_product_info_wrapper(product_name: str) -> dict:
        return get_product_info(seller_id=seller_id, product_name=product_name)

    def track_order_wrapper(order_id: str) -> dict:
        return track_order(order_id=order_id)

    def place_order_wrapper(items: List[dict]) -> dict:
        # Validate items manually
        for item in items:
            if not isinstance(item, dict) or "product_id" not in item or "quantity" not in item:
                raise ValueError("Each item must be a dict with 'product_id' (str) and 'quantity' (int)")
            if not isinstance(item["product_id"], str) or not isinstance(item["quantity"], int):
                raise ValueError("product_id must be a string and quantity must be an integer")
        return place_order(seller_id=seller_id, user_id=user_id, items=items)

    def save_user_wrapper(name: str, email: str, address: str, number: str) -> dict:
        return save_user(user_id=user_id, name=name, email=email, address=address, number=number)

    def get_user_info_wrapper(*args, **kwargs) -> dict:
        # Ignore any passed arguments since this function doesn't need them
        return get_user_info(user_id=user_id)

    def check_user_exists_wrapper(*args, **kwargs) -> bool:
        # Ignore any passed arguments since this function doesn't need them
        return check_user_exists(user_id=user_id)

    def get_all_products_wrapper(*args, **kwargs) -> List[str]:
        return get_all_products(seller_id=seller_id)

    def update_user_info_wrapper(name: str = "", email: str = "", address: str = "", number: str = "") -> dict:
        # Convert empty strings to None for compatibility
        name = None if not name else name
        email = None if not email else email
        address = None if not address else address
        number = None if not number else number
        return update_user_info(user_id=user_id, name=name, email=email, address=address, number=number)

    # Create tools with clear, LLM-friendly descriptions
    function_tools = [
        StructuredTool(
            name="get_product_info",
            func=get_product_info_wrapper,
            description="Retrieves details for a product by its name. Requires a product name (string).",
            args_schema=GetProductInfoInput
        ),
        StructuredTool(
            name="track_order",
            func=track_order_wrapper,
            description="Tracks the status of an order by its order ID. Requires an order ID (string).",
            args_schema=TrackOrderInput
        ),
        StructuredTool(
            name="place_order",
            func=place_order_wrapper,
            description="Places an order for multiple products. Requires a list of items, each with a product_id (string) and quantity (integer).",
            args_schema=PlaceOrderInput
        ),
        StructuredTool(
            name="save_user",
            func=save_user_wrapper,
            description="Creates a new user with the provided details. Requires name (string), email (string), address (string), and phone number (string).",
            args_schema=SaveUserInput
        ),
        StructuredTool(
            name="get_user_info",
            func=get_user_info_wrapper,
            description="Retrieves information for the current user.",
            args_schema=EmptyInput
        ),
        StructuredTool(
            name="check_user_exists",
            func=check_user_exists_wrapper,
            description="Checks if the current user exists in the system.",
            args_schema=EmptyInput
        ),
        StructuredTool(
            name="update_user_info",
            func=update_user_info_wrapper,
            description="Updates information for the current user. All parameters are optional: name (string), email (string), address (string), phone number (string). Provide empty strings for unchanged fields.",
            args_schema=UpdateUserInfoInput
        ),
        StructuredTool(
            name="get_all_products",
            func=get_all_products_wrapper,
            description="Retrieves all products for the current seller. Returns a list of product details as strings.",
            args_schema=EmptyInput
        )
    ]

    # Log tool creation and schemas for debugging
    logger.debug(f"[Function Agent] Created tools: {[tool.name for tool in function_tools]}")
    for tool in function_tools:
        try:
            schema = tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else "No schema"
            logger.debug(f"[Function Agent] Tool {tool.name} schema: {json.dumps(schema, indent=2)}")
        except Exception as e:
            logger.error(f"[Function Agent] Error serializing schema for tool {tool.name}: {str(e)}")

    # Create LLM with tools bound, with error handling
    try:
        llm_with_tools = llm.bind_tools(function_tools)
        logger.debug("[Function Agent] Tools successfully bound to LLM")
    except Exception as e:
        logger.error(f"[Function Agent] Error binding tools to LLM: {str(e)}")
        raise

    # Create prompt template with chat history support
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a function execution agent for seller ID: """+ seller_id+ """, user ID: """+user_id+""".
            
            Based on the user's intent and input, execute the appropriate tools to fulfill the request.
            
            Available tools: {{[tool.name for tool in function_tools]}}
            
            Use these examples to guide your actions: {{examples}}
            Use chat history to maintain context from previous interactions.
            
            Tool descriptions:
            - get_product_info: Retrieves product details (requires: product_name as string)
            - track_order: Tracks order status (requires: order_id as string)
            - place_order: Places an order (requires: items as list of {{product_id: string, quantity: integer}})
            - save_user: Creates a new user (requires: name, email, address, number as strings)
            - get_user_info: Retrieves user information (no parameters)
            - check_user_exists: Checks if user exists (no parameters)
            - update_user_info: Updates user information (optional: name, email, address, number as strings; use empty strings for unchanged fields)
            
            For intent: {{intent}}, select and execute the appropriate tool(s).
            
            IMPORTANT:
            - Execute tools directly; do not describe actions without executing.
            - For place_order, verify user existence with check_user_exists, then get_user_info or save_user as needed.
            - Extract parameters accurately from user input and chat history.
            - Provide clear responses based on tool outputs.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent with tools
    try:
        agent = create_openai_tools_agent(llm_with_tools, function_tools, prompt)
        logger.debug("[Function Agent] Agent created successfully")
    except Exception as e:
        logger.error(f"[Function Agent] Error creating agent: {str(e)}")
        raise

    executor = AgentExecutor(agent=agent, tools=function_tools, verbose=True)
    
    def invoke_function_chain(input_data: dict, chat_history: list | None = None) -> dict:
        """
        Invoke the unified function agent with LLM-driven tool execution.

        Args:
            input_data (dict): Dictionary containing 'input' and 'intent' keys.
            chat_history (list): List of previous messages for context.

        Returns:
            dict: Response with function execution results.
        """
        if chat_history is None:
            chat_history = []
            
        logger.info(f"[Function Agent] Invoking unified function agent with input: {input_data}")
        logger.debug(f"[Function Agent] Input data: {input_data}")

        try:
            # Get RAG examples
            examples = get_rag_examples(input_data["input"], seller_id, k=3)
            
            # Convert chat history to proper format for ChatPromptTemplate
            formatted_chat_history = []
            for msg in chat_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    if msg["role"] == "user":
                        formatted_chat_history.append(("human", msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_chat_history.append(("assistant", msg["content"]))

            # Execute the agent
            result = executor.invoke({
                "input": input_data["input"],
                "intent": input_data["intent"],
                "examples": examples,
                "chat_history": formatted_chat_history
            })

            logger.info(f"[Function Agent] Agent execution completed")
            logger.debug(f"[Function Agent] Agent result: {result}")

            # Extract the output
            agent_output = result.get("output", "No output available")
            
            # Log the query for analytics
            try:
                log_query(
                    query=input_data["input"],
                    intent=input_data["intent"],
                    entities="LLM-extracted",
                    response=str(agent_output),
                    seller_id=seller_id,
                    user_id=user_id
                )
                logger.debug("[Function Agent] Query logged successfully")
            except Exception as e:
                logger.error(f"[Function Agent] Error logging query: {str(e)}")

            return {
                "function": "agent_execution",
                "output": agent_output,
                "success": True
            }

        except Exception as e:
            logger.error(f"[Function Agent] Error in agent execution: {str(e)}")
            return {
                "function": "error",
                "output": f"Error in function execution: {str(e)}",
                "error": str(e),
                "success": False
            }

    logger.info("[Function Agent] Unified function agent created successfully")
    return invoke_function_chain

def create_response_agent(seller_id):
    logger.info(f"[Response Agent] Creating response agent for seller_id: {seller_id}")
    
    # First chain: Analyze context and determine response strategy
    strategy_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Analyze the context for seller ID: """ + seller_id + """ and determine the best response strategy.
            
            Intent: {{intent}}
            Function Output: {{function_output}}
            User Input: {{input}}
            
            Consider:
            1. Was the function successful?
            2. Does the user need additional information?
            3. Should we ask follow-up questions?
            4. Is there an error that needs explanation?
            5. Is the user input ambiguous or unclear?
            6. Are there missing details needed to complete the request?
            
            Available response strategies:
            - success: Function executed successfully, provide helpful response
            - error: Function failed, explain the error and suggest solutions
            - follow_up: Ask for additional information needed
            - clarification: Input is ambiguous, ask clarifying questions
            - information: Provide general information without function execution
            
            Respond with JSON: {{"strategy": "success|error|follow_up|clarification|information", "tone": "helpful|apologetic|informative", "include_suggestions": true/false, "clarification_needed": ["<info1>", "<info2>"] if strategy is clarification}}
        """),
        ("human", "{input}")
    ])
    
    # Second chain: Generate the actual response
    response_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            Generate a response for seller ID: {seller_id} using the strategy: {{strategy_analysis}}
            
            Context:
            - User Input: {{input}}
            - Intent: {{intent}}
            - Function Output: {{function_output}}
            - Dataset Response: {{dataset_response}}
            
            Guidelines based on strategy:
            1. SUCCESS: Be conversational and helpful, include relevant information from function output
            2. ERROR: Apologetically explain the error and suggest solutions
            3. FOLLOW_UP: Ask specific questions to gather needed information
            4. CLARIFICATION: Ask clear, specific questions to resolve ambiguity
            5. INFORMATION: Provide helpful information using dataset response if available
            
            Additional guidelines:
            - Use dataset response as reference if relevant
            - Follow the determined strategy and tone
            - Add suggestions if recommended by strategy
            - For clarification requests, ask one clear question at a time
            - Be specific about what information is needed
            
            IMPORTANT: Return ONLY the actual response message that should be shown to the user. 
            Do not include any explanations, metadata, or additional context. 
            Just provide the direct response text.
        """),
        ("human", "Generate response based on the above context.")
    ])
    
    # Chain them together
    def response_chain(input_data):
        logger.info(f"[Response Agent] Processing response for seller_id: {seller_id}")
        logger.debug(f"[Response Agent] Input data: {input_data}")
        
        try:
            # Step 1: Determine strategy
            logger.info("[Response Agent] Step 1: Determining response strategy")
            strategy_result = (
                {
                    "input": RunnablePassthrough(),
                    "intent": lambda x: x["intent"],
                    "function_output": lambda x: x["function_output"]
                }
                | strategy_prompt
                | llm
                | StrOutputParser()
            ).invoke(input_data)
            logger.debug(f"[Response Agent] Strategy result: {strategy_result}")
            
            # Step 2: Generate response
            logger.info("[Response Agent] Step 2: Generating response")
            try:
                dataset_response = vector_store.similarity_search(input_data["input"], k=1)[0].metadata.get("response", "")
                logger.debug(f"[Response Agent] Dataset response found: {dataset_response}")
            except Exception as e:
                logger.warning(f"[Response Agent] Could not fetch dataset response: {str(e)}")
                dataset_response = ""
                
            response_result = (
                {
                    "input": RunnablePassthrough(),
                    "intent": lambda x: x["intent"],
                    "function_output": lambda x: x["function_output"],
                    "dataset_response": lambda _: dataset_response,
                    "strategy_analysis": lambda _: strategy_result
                }
                | response_generation_prompt
                | llm
                | StrOutputParser()
            ).invoke(input_data)
            logger.info(f"[Response Agent] Response generation completed")
            logger.debug(f"[Response Agent] Final response: {response_result}")
            
            # Clean up the response - remove any extra formatting or explanations
            cleaned_response = response_result.strip()
            
            # If response starts with explanatory text, try to extract just the response
            if "**Response:**" in cleaned_response:
                cleaned_response = cleaned_response.split("**Response:**")[1].strip()
            elif "Response:" in cleaned_response:
                cleaned_response = cleaned_response.split("Response:")[1].strip()
            
            # Remove any trailing explanatory text or dashes
            if "---" in cleaned_response:
                cleaned_response = cleaned_response.split("---")[0].strip()
            
            # Remove quotes if the entire response is wrapped in quotes
            if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                cleaned_response = cleaned_response[1:-1].strip()
            
            logger.debug(f"[Response Agent] Cleaned response: {cleaned_response}")
            
            return cleaned_response
        except Exception as e:
            logger.error(f"[Response Agent] Error in response generation: {str(e)}")
            raise
    
    logger.info("[Response Agent] Response agent created successfully")
    return response_chain



# Multi-Agent System
def create_multi_agent_system(seller_id: str, user_id: str):
    logger.info(f"[Multi-Agent System] Creating multi-agent system for seller_id: {seller_id}, user_id: {user_id}")
    
    # Initialize persistent chat history
    chat_history = []
    
    try:
        intent_agent = create_intent_agent(seller_id)
        function_agent = create_function_agent(seller_id, user_id)
        response_agent = create_response_agent(seller_id)
        
        logger.info("[Multi-Agent System] All agents created successfully")
    except Exception as e:
        logger.error(f"[Multi-Agent System] Error creating agents: {str(e)}")
        raise

    def process_input(input_data, external_chat_history: list | None = None):
        nonlocal chat_history
        
        # Use external chat history if provided, otherwise use internal
        if external_chat_history is not None:
            chat_history = external_chat_history
            
        logger.info(f"[Multi-Agent System] Starting processing for seller_id: {seller_id}, user_id: {user_id}")
        logger.debug(f"[Multi-Agent System] Input data: {input_data}")
        logger.debug(f"[Multi-Agent System] Chat history length: {len(chat_history)}")
        
        try:
            user_input = input_data["input"]
            logger.info(f"[Multi-Agent System] Processing user input: {user_input}")
            
            # Add user input to chat history
            chat_history.append({"role": "user", "content": user_input})
            
            # Step 1: Intent Detection
            logger.info("[Multi-Agent System] Step 1: Detecting intent")
            intent_result = intent_agent(user_input)
            logger.debug(f"[Multi-Agent System] Intent result: {intent_result}")
            
            # Parse the JSON response safely
            intent_data = parse_llm_json_response(intent_result)
            if not intent_data or "intent" not in intent_data:
                logger.error(f"[Multi-Agent System] Failed to parse intent result")
                return "I'm sorry, I couldn't understand your request. Please try again."
            
            intent = intent_data["intent"]
            logger.info(f"[Multi-Agent System] Detected intent: {intent}")
            
            # Step 2: Function Execution (LLM handles tool execution)
            logger.info("[Multi-Agent System] Step 2: Executing function via LLM agent")
            function_result = function_agent({"input": user_input, "intent": intent}, chat_history)
            
            if not function_result.get("success", False):
                logger.error(f"[Multi-Agent System] Function execution failed: {function_result.get('error', 'Unknown error')}")
                return function_result.get("output", "I encountered an error while processing your request.")
            
            # Extract function output from result
            function_output = function_result.get("output", "No output available")
            logger.info(f"[Multi-Agent System] Function executed successfully via LLM agent")
            
            # Step 3: Response Generation
            logger.info("[Multi-Agent System] Step 3: Generating response")
            response = response_agent({
                "input": user_input,
                "intent": intent,
                "function_output": function_output
            })
            
            # Add assistant response to chat history
            chat_history.append({"role": "assistant", "content": response})
            
            # Limit chat history to prevent token overflow (keep last 10 exchanges)
            if len(chat_history) > 20:  # 10 user + 10 assistant messages
                chat_history = chat_history[-20:]
                
            logger.info("[Multi-Agent System] Processing completed successfully")
            logger.debug(f"[Multi-Agent System] Final response: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(f"[Multi-Agent System] {error_msg}")
            logger.exception("[Multi-Agent System] Exception details:")
            
            # Add error to chat history
            chat_history.append({"role": "assistant", "content": error_msg})
            
            return error_msg

    return {"executor": process_input, "chat_history": chat_history}


# def create_user_management_agent(seller_id, user_id):
#     logger.info(f"[User Management Agent] Creating user management agent for seller_id: {seller_id}, user_id: {user_id}")
    
#     tools_with_context = [
#         Tool(
#             name=t.name,
#             func=t.func,
#             description=t.description.format(seller_id=seller_id, user_id=user_id)
#         ) for t in tools if t.name in ["save_user", "get_user_info", "check_user_exists", "update_user_info"]
#     ]
    
#     logger.debug(f"[User Management Agent] Created {len(tools_with_context)} user management tools")
    
#     llm_with_tools = llm.bind_tools(tools_with_context)
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", f"""
#             Manage user details for seller ID: {seller_id}, user ID: {user_id}.
#             - Check if user exists using check_user_exists
#             - If user doesn't exist and order is requested, ask for details and use save_user
#             - If user exists, confirm details with get_user_info
#             - Update details with update_user_info if requested
#             Respond with JSON: {{"action": "<action>", "response": "<response>"}}
#         """),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}")
#     ])
    
#     agent = create_openai_tools_agent(llm_with_tools, tools_with_context, prompt)
#     executor = AgentExecutor(agent=agent, tools=tools_with_context, verbose=True)
    
#     logger.info("[User Management Agent] User management agent created successfully")
#     return executor


    # agent = create_openai_tools_agent(llm, tools, prompt)
    # return AgentExecutor(agent=agent, tools=tools, verbose=True)

# # Default agent executor (will be replaced by create_agent_executor)
# tools = [
#     Tool(name="get_product_info", func=get_product_info, description="Get product details by name and seller ID. Required parameters: product_name (string), seller_id (string)"),
#     Tool(name="track_order", func=track_order, description="Track order status by order ID. Required parameters: order_id (string)"),
#     Tool(name="place_order", func=place_order, description="Place an order for multiple products. Required parameters: seller_id (string), user_id (string), items (list of dictionaries with product_id and quantity keys)"),
#     Tool(name="log_query", func=log_query, description="Log user queries with intent, entities, response, seller ID and user ID. Required parameters: query (string), intent (string), entities (string), response (string), seller_id (string), user_id (string)"),
#     # Tool(name="query_context", func=query_context, description="Search product/FAQ documents using RAG. Required parameters: query (string), seller_id (string)")
# ]

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a business assistant for seller ID: {seller_id}. Use tools to fetch product info, track orders, or place orders. Always include the seller_id parameter when calling tools. Log all queries. For general questions, use RAG context if relevant."),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad")
# ])

# agent = create_openai_tools_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)