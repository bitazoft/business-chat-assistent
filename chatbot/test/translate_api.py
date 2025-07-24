from google.cloud import translate_v2 as translate
import os

# Set up Google Cloud credentials
# Replace 'path/to/your/service-account-key.json' with your actual key file path
# Alternatively, set the environment variable: export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\\Project\\business-chat-assistent\\chatbot\\config\\rapid-strength-394704-b813738d1cb9.json"

def translate_text(text, target_language="es"):
    """Translate text to the target language using Google Cloud Translation API."""
    client = translate.Client()
    languages = client.get_languages()
    # for lang in languages:
    #     print(f"Language: {lang['name']} ({lang['language']})")
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]

def chatbot_response(user_input, target_language="es"):
    """Generate a chatbot response and translate it to the target language."""
    # Simple chatbot response logic (replace with your LLM logic if needed)
    response = f"{user_input}"
    
    # Translate the response to the target language
    translated_response = translate_text(response, target_language)
    return translated_response

# Example usage
if __name__ == "__main__":
    # Simulate user input
    user_input = "hello, how are you?"
    target_language = "si"  # Spanish; change to any language code (e.g., "fr" for French, "hi" for Hindi)
    
    # Get and print translated chatbot response
    response = chatbot_response(user_input, target_language)
    print(f"User input: {user_input}")
    print(f"Chatbot response (in {target_language}): {response}")

    user_input = "මගේ උපකාරය ලබා ගැනීමට සුබ පැතුම්"
    target_language = "en"  # French

    response = chatbot_response(user_input, target_language)
    print(f"User input: {user_input}")
    print(f"Chatbot response (in {target_language}): {response}")