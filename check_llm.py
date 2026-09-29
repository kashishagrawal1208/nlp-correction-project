"""
check_llm.py: temporary script to test the Gemini connection.
Run from the project root:  python check_llm.py
"""

import os

import yaml
from dotenv import load_dotenv
from google import genai

# 1. Load the key from the .env file into the environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key == "your_api_key_here":
    print("ERROR: No real API key found. Check your .env file.")
    raise SystemExit(1)

print("API key found (length:", len(api_key), "characters)")

# 2. Create a client: the object that talks to Google's servers
client = genai.Client(api_key=api_key)

# 3. List the models this key can use
print("\nModels available to your key (that can generate text):")
try:
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions:
            print("  ", model.name.replace("models/", ""))
except Exception as error:
    print("Could not list models:", error)

# 4. Read the model name from config.yaml and send one small request
with open("config/config.yaml", "r", encoding="utf-8") as file:
    model_name = yaml.safe_load(file)["llm"]["model"]

print(f"\nTesting model from config.yaml: {model_name}")
try:
    response = client.models.generate_content(
        model=model_name,
        contents="Reply with exactly these two words: connection working",
    )
    print("LLM replied:", response.text)
    print("\nSUCCESS. The API connection works.")
except Exception as error:
    print("REQUEST FAILED:", error)
    print("\nIf the error says the model was not found, pick a name from the")
    print("list above and put it in config/config.yaml under llm: model:")