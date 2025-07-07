import requests

# Chat session configuration
session_id = "session_12345"
seller_id = "1"
user_id = "user3"

# Initialize chat history
chat_history = []

# Chat endpoint
url = "http://127.0.0.1:8000/chat"

print("🧠 Chatbot session started. Type 'exit' to quit.\n")

# Main chat loop
while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("👋 Chat ended.")
        break

    # Append user message to history
    chat_history.append({"role": "user", "content": user_input})

    # Prepare payload
    payload = {
        "message": user_input,
        "session_id": session_id,
        "seller_id": seller_id,
        "user_id": user_id,
        "chat_history": chat_history
    }

    # Send request
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        assistant_reply = response.json().get("response", "[No response returned]")
    except Exception as e:
        assistant_reply = f"[Error contacting server: {e}] \n {response.json()}"

    # Print and store assistant reply
    print(f"Assistant: {assistant_reply}\n")
    chat_history.append({"role": "assistant", "content": assistant_reply})
