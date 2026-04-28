import os
from dotenv import load_dotenv
 
load_dotenv()
 
class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL: str = "llama-3.3-70b-versatile"
    APP_NAME: str = "BizMind"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
 
config = Config()
print(f"🔑 Groq Key loaded: {config.GROQ_API_KEY[:12]}...")
print(f"🤖 Model: {config.MODEL}")
