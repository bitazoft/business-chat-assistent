from langchain_openai import ChatOpenAI
from langchain_deepseek.chat_models import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from db.database import SessionLocal
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
    create_tmp_user_id
)
from vector_store.vector_store import vector_store
import os
import numpy as np
import requests
from datetime import datetime
from typing import List
from functools import partial

# Load environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# Make functions available for import
__all__ = ['create_agent_executor', 'log_query', 'get_product_info', 'track_order', 'place_order', 'get_user_info', 'update_user_info', 'check_user_exists', 'save_user', 'create_tmp_user_id']

# LangChain Agent Setup
llm = ChatDeepSeek(
    model="deepseek-chat",  # Replace with actual DeepSeek model name
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE
)

def create_agent_executor(seller_id: str, user_id: str):
    """Create an agent executor with seller_id and user_id bound to tools"""
    
    # Create tools with bound seller_id and user_id
    tools = [
        Tool(
            name="get_product_info", 
            func=partial(get_product_info, seller_id=seller_id), 
            description="Get product details by name. Required parameters: product_name (string)"
        ),
        Tool(
            name="track_order", 
            func=track_order, 
            description="Track order status by order ID. Required parameters: order_id (string)"
        ),
        Tool(
            name="place_order", 
            func=partial(place_order, seller_id=seller_id, user_id=user_id), 
            description="Place an order for multiple products. Required parameters: items (list of dictionaries with product_id and quantity keys). product_id can be either numeric ID or product name string."
        ),
        Tool(
            name="log_query", 
            func=partial(log_query, seller_id=seller_id, user_id=user_id), 
            description="Log user queries with intent, entities, and response. Required parameters: query (string), intent (string), entities (string), response (string)"
        ),
        Tool(
            name="save_user",
            func=partial(save_user, user_id=user_id),
            description="Create a new user with provided details. Required parameters: name (string), email (string), address (string), number (string)"
        ),
        Tool(
            name="get_user_info",
            func=partial(get_user_info, user_id=user_id),
            description="Retrieve user information from database. No additional parameters required - user_id is automatically provided."
        ),
        Tool(
            name="check_user_exists",
            func=partial(check_user_exists, user_id=user_id),
            description="Check if user exists in database. No additional parameters required - user_id is automatically provided."
        ),
        Tool(
            name="update_user_info",
            func=partial(update_user_info, user_id=user_id),
            description="Update user information in database. Parameters: name (string, optional), email (string, optional), address (string, optional), number (string, optional). Only provided parameters will be updated."
        )
        # Tool(name="query_context", func=partial(query_context, seller_id=seller_id), description="Search product/FAQ documents using RAG. Required parameters: query (string)")
    ]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a business assistant for seller ID: {seller_id}. Use tools to fetch product info, track orders, or place orders. The seller_id and user_id are automatically provided to tools that need them. 

        USER MANAGEMENT WORKFLOW:
        - For general inquiries (product info, order tracking, questions): No need to collect user details
        - ONLY when user wants to PLACE AN ORDER:
          1. Check if user exists using check_user_exists
          2. If user doesn't exist, ask for their details (name, email, address, phone number) and create them using save_user
          3. If user exists, get their info using get_user_info and show it to confirm details are correct
          4. If user wants to update any information, use update_user_info with only the fields they want to change
          5. Only proceed with placing order after confirming user details

        GENERAL INSTRUCTIONS:
        - Log all queries using log_query
        - For product questions, use get_product_info (no user details needed)
        - For order tracking, use track_order (no user details needed)
        - For placing orders, use place_order (user details required first)
        - Always be helpful and only ask for information when necessary"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

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