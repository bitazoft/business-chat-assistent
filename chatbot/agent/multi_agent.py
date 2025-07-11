from functools import partial
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_deepseek import DeepSeekEmbeddings
import pandas as pd
import os

# Environment variables
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

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

# Placeholder tool functions (replace with your actual implementations)
def get_product_info(product_name: str, seller_id: str):
    return f"Product info for {product_name} from seller {seller_id}"

def track_order(order_id: str):
    return f"Order status for {order_id}"

def place_order(items: list, seller_id: str, user_id: str):
    return f"Order placed for {items} by user {user_id} for seller {seller_id}"

def log_query(query: str, intent: str, entities: str, response: str, seller_id: str, user_id: str):
    return f"Logged query: {query}, Intent: {intent}, Entities: {entities}, Response: {response}"

def save_user(name: str, email: str, address: str, number: str, user_id: str):
    return f"User {user_id} saved with name: {name}, email: {email}"

def get_user_info(user_id: str):
    return f"User info for {user_id}"

def check_user_exists(user_id: str):
    return True  # Replace with actual check

def update_user_info(user_id: str, name: str = None, email: str = None, address: str = None, number: str = None):
    return f"Updated user {user_id} with provided fields"

def create_tmp_user_id():
    return "tmp_user_123"  # Replace with actual logic

# Load vector store
df = pd.read_csv("your_dataset.csv")
embeddings = DeepSeekEmbeddings(model="deepseek-chat", api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
unique_intents = df["intent"].unique().tolist()

# Initialize LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
    temperature=0.3,
    json_mode=True
)

# Define tools
tools = [
    Tool(
        name="get_product_info",
        func=partial(get_product_info, seller_id="{seller_id}"),
        description="Get product details by name. Required parameters: product_name (string)"
    ),
    Tool(
        name="track_order",
        func=track_order,
        description="Track order status by order ID. Required parameters: order_id (string)"
    ),
    Tool(
        name="place_order",
        func=partial(place_order, seller_id="{seller_id}", user_id="{user_id}"),
        description="Place an order for multiple products. Required parameters: items (list of dictionaries with product_id and quantity keys)"
    ),
    Tool(
        name="log_query",
        func=partial(log_query, seller_id="{seller_id}", user_id="{user_id}"),
        description="Log user queries with intent, entities, and response. Required parameters: query (string), intent (string), entities (string), response (string)"
    ),
    Tool(
        name="save_user",
        func=partial(save_user, user_id="{user_id}"),
        description="Create a new user. Required parameters: name (string), email (string), address (string), number (string)"
    ),
    Tool(
        name="get_user_info",
        func=partial(get_user_info, user_id="{user_id}"),
        description="Retrieve user information. No additional parameters required."
    ),
    Tool(
        name="check_user_exists",
        func=partial(check_user_exists, user_id="{user_id}"),
        description="Check if user exists. No additional parameters required."
    ),
    Tool(
        name="update_user_info",
        func=partial(update_user_info, user_id="{user_id}"),
        description="Update user information. Parameters: name (string, optional), email (string, optional), address (string, optional), number (string, optional)"
    )
]

# Helper function for RAG
def get_rag_examples(user_input, seller_id, k=3):
    results = vector_store.similarity_search(user_input, k=k)
    return "\n".join([
        f"Instruction: {result.page_content}, Intent: {result.metadata['intent']}, Response: {result.metadata['response']}"
        for result in results if result.metadata.get('category') == seller_id or not result.metadata.get('category')
    ])

# Agent Definitions
def create_intent_agent(seller_id):
    # First chain: Extract key entities and context
    entity_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Extract key entities and context from the user input for seller ID: {seller_id}.
            Look for: product names, order IDs, user information, quantities, etc.
            
            Examples: {examples}
            
            Respond with JSON: {"entities": {"products": [], "order_ids": [], "user_info": {}, "other": []}, "context": "<summary>"}
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
            
            Respond with JSON: {"intent": "<intent>", "confidence": 0.0-1.0, "reasoning": "<explanation>"}
        """),
        ("human", "Original input: {input}")
    ])
    
    # Chain them together
    def intent_chain(input_data):
        # Step 1: Extract entities
        entity_result = (
            {"input": RunnablePassthrough(), "examples": lambda x: get_rag_examples(x["input"], seller_id)}
            | entity_extraction_prompt
            | llm
            | StrOutputParser()
        ).invoke(input_data)
        
        # Step 2: Classify intent
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
        
        return intent_result
    
    return intent_chain

def create_function_agent(seller_id, user_id):
    tools_with_context = [
        Tool(
            name=t.name,
            func=t.func,
            description=t.description.format(seller_id=seller_id, user_id=user_id)
        ) for t in tools
    ]
    llm_with_tools = llm.bind_tools(tools_with_context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            You are a function execution agent for seller ID: {seller_id}. Use tools to execute the intent: {{intent}}.
            Log all actions using log_query. Respond with JSON: {{"function": "<function_name>", "parameters": {{parameters}}}}
        """),
        ("human", "{input}")
    ])
    chain = (
        {"input": RunnablePassthrough(), "intent": lambda x: x["intent"]}
        | prompt
        | llm_with_tools
        | StrOutputParser()
    )
    return chain

def create_user_management_agent(seller_id, user_id):
    tools_with_context = [
        Tool(
            name=t.name,
            func=t.func,
            description=t.description.format(seller_id=seller_id, user_id=user_id)
        ) for t in tools if t.name in ["save_user", "get_user_info", "check_user_exists", "update_user_info"]
    ]
    llm_with_tools = llm.bind_tools(tools_with_context)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            Manage user details for seller ID: {seller_id}, user ID: {user_id}.
            - Check if user exists using check_user_exists
            - If user doesn't exist and order is requested, ask for details and use save_user
            - If user exists, confirm details with get_user_info
            - Update details with update_user_info if requested
            Respond with JSON: {{"action": "<action>", "response": "<response>"}}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    agent = create_openai_tools_agent(llm_with_tools, tools_with_context, prompt)
    return AgentExecutor(agent=agent, tools=tools_with_context, verbose=True)

def create_response_agent(seller_id):
    # First chain: Analyze context and determine response strategy
    strategy_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Analyze the context for seller ID: {seller_id} and determine the best response strategy.
            
            Intent: {intent}
            Function Output: {function_output}
            
            Consider:
            1. Was the function successful?
            2. Does the user need additional information?
            3. Should we ask follow-up questions?
            4. Is there an error that needs explanation?
            
            Respond with JSON: {"strategy": "success|error|follow_up|clarification", "tone": "helpful|apologetic|informative", "include_suggestions": true/false}
        """.format(seller_id=seller_id)),
        ("human", "{input}")
    ])
    
    # Second chain: Generate the actual response
    response_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Generate a response for seller ID: {seller_id} using the strategy: {strategy_analysis}
            
            Context:
            - User Input: {input}
            - Intent: {intent}
            - Function Output: {function_output}
            - Dataset Response: {dataset_response}
            
            Guidelines:
            1. Be conversational and helpful
            2. Include relevant information from function output
            3. Use dataset response as reference if relevant
            4. Follow the determined strategy and tone
            5. Add suggestions if recommended by strategy
            
            Generate a natural, helpful response.
        """.format(seller_id=seller_id)),
        ("human", "Generate response based on the above context.")
    ])
    
    # Chain them together
    def response_chain(input_data):
        # Step 1: Determine strategy
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
        
        # Step 2: Generate response
        try:
            dataset_response = vector_store.similarity_search(input_data["input"], k=1)[0].metadata.get("response", "")
        except:
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
        
        return response_result
    
    return response_chain

def create_clarification_agent(seller_id):
    # First chain: Analyze ambiguity
    ambiguity_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Analyze if the user input is ambiguous or unclear for seller ID: {seller_id}.
            Consider these factors:
            1. Missing key information
            2. Multiple possible interpretations
            3. Unclear intent from available intents: {intents}
            
            Respond with JSON: {"is_ambiguous": true/false, "reason": "<explanation>", "missing_info": ["<info1>", "<info2>"]}
        """.format(seller_id=seller_id)),
        ("human", "{input}")
    ])
    
    # Second chain: Generate clarification questions
    clarification_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            Based on the ambiguity analysis: {analysis}
            Generate a helpful clarifying question for seller ID: {seller_id}.
            
            Make the question:
            1. Specific to the missing information
            2. Easy to understand
            3. Actionable for the user
            
            Respond with JSON: {"clarification": "<question>", "suggested_options": ["<option1>", "<option2>"]}
        """.format(seller_id=seller_id)),
        ("human", "Original input: {input}")
    ])
    
    # Chain them together
    def clarification_chain(input_data):
        # Step 1: Analyze ambiguity
        ambiguity_analysis = (
            {"input": RunnablePassthrough(), "intents": lambda _: ", ".join(unique_intents)}
            | ambiguity_prompt
            | llm
            | StrOutputParser()
        ).invoke(input_data)
        
        # Step 2: Generate clarification if needed
        analysis_result = eval(ambiguity_analysis)
        if analysis_result["is_ambiguous"]:
            clarification_result = (
                {"input": RunnablePassthrough(), "analysis": lambda _: ambiguity_analysis}
                | clarification_prompt
                | llm
                | StrOutputParser()
            ).invoke(input_data)
            return clarification_result
        else:
            return '{"clarification": null, "proceed": true}'
    
    return clarification_chain

# Multi-Agent System
def create_multi_agent_system(seller_id: str, user_id: str):
    intent_agent = create_intent_agent(seller_id)
    function_agent = create_function_agent(seller_id, user_id)
    user_management_agent = create_user_management_agent(seller_id, user_id)
    response_agent = create_response_agent(seller_id)
    clarification_agent = create_clarification_agent(seller_id)

    # Function mapping for tool execution
    function_map = {
        "get_product_info": partial(get_product_info, seller_id=seller_id),
        "track_order": track_order,
        "place_order": partial(place_order, seller_id=seller_id, user_id=user_id),
        "log_query": partial(log_query, seller_id=seller_id, user_id=user_id),
        "save_user": partial(save_user, user_id=user_id),
        "get_user_info": partial(get_user_info, user_id=user_id),
        "check_user_exists": partial(check_user_exists, user_id=user_id),
        "update_user_info": partial(update_user_info, user_id=user_id)
    }

    def process_input(input_data, chat_history=[]):
        try:
            user_input = input_data["input"]
            
            # Step 1: Check for clarification needed
            clarification_result = clarification_agent(user_input)
            clarification_data = eval(clarification_result)
            
            if clarification_data.get("clarification"):
                return clarification_data["clarification"]
            
            # Step 2: Intent Detection with enhanced chain
            intent_result = intent_agent(user_input)
            intent_data = eval(intent_result)
            intent = intent_data["intent"]
            
            # Step 3: User Management (for place_order)
            if intent == "place_order":
                user_check = user_management_agent.invoke({"input": user_input, "chat_history": chat_history})
                user_response = eval(user_check["output"])["response"]
                if "ask for details" in user_response.lower():
                    return user_response  # Prompt for user details
                chat_history.append(user_check)

            # Step 4: Function Execution
            function_result = function_agent.invoke({"input": user_input, "intent": intent})
            function_data = eval(function_result)
            
            # Execute the actual function if identified
            function_output = function_result
            if function_data.get("function") and function_data["function"] in function_map:
                try:
                    params = function_data.get("parameters", {})
                    if isinstance(params, dict) and "params" in params:
                        params = params["params"]
                    
                    # Execute the function
                    if params:
                        function_output = function_map[function_data["function"]](**params)
                    else:
                        function_output = function_map[function_data["function"]]()
                except Exception as e:
                    function_output = f"Error executing function: {str(e)}"
            
            # Step 5: Log Query
            log_query(user_input, intent, str(intent_data.get("entities", {})), str(function_output), seller_id, user_id)

            # Step 6: Response Generation with enhanced chain
            response = response_agent({
                "input": user_input,
                "intent": intent,
                "function_output": function_output
            })
            
            return response
            
        except Exception as e:
            return f"Error processing request: {str(e)}"

    return {"executor": process_input, "chat_history": []}

# Example usage
if __name__ == "__main__":
    system = create_multi_agent_system(seller_id="seller_123", user_id="user_123")
    executor = system["executor"]
    chat_history = system["chat_history"]
    response = executor({"input": "Book a meeting for tomorrow at 10 AM with Alice"}, chat_history)
    print(response)