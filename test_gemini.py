import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL")

print(f"Testing Gemini API with model: {model_name}")
print(f"API Key starts with: {api_key[:10]}...")

client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model=model_name,
        contents=["hi"]
    )
    print("Success!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error Type: {type(e)}")
    print(f"Error Message: {e}")
