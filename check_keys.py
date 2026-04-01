from dotenv import load_dotenv
import os
load_dotenv()
gk = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
ok = os.getenv("OPENAI_API_KEY")
print("Gemini:", "OK" if gk else "MISSING")
print("OpenAI:", "OK" if ok else "MISSING")
