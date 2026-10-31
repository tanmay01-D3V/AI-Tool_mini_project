import os
import sys
from dotenv import load_dotenv
from google import genai

PLACEHOLDER = "your_key_here"


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip() or api_key.strip() == PLACEHOLDER:
        print("GEMINI_API_KEY is missing or still set to the placeholder in .env. Please configure your API key in .env.")
        sys.exit(1)

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly one word: OK",
        )
        if response and response.text:
            print(response.text.strip())
        else:
            print("Received empty response from Gemini API.")
    except Exception as e:
        err_str = str(e)
        if api_key in err_str:
            err_str = err_str.replace(api_key, "[REDACTED]")
        print(f"Error communicating with Gemini API: {err_str}")
        sys.exit(1)


if __name__ == "__main__":
    main()

