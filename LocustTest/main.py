from fastapi import FastAPI, HTTPException
from groq import Groq
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class TextOutput(BaseModel):
    response: str



GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")

@lru_cache(maxsize=1)
def get_groq_client():
    return Groq(api_key=GROQ_API_KEY)

system_prompt = (
    "You are an AI assistant with a witty and humorous personality. "
    "You're helpful, engaging, and intelligent—but always respectful. "
    "Use clever, light humor where appropriate, but never be sarcastic, offensive, or biased. "
    "Stay factual, avoid speculation, and respond in clear, friendly language. "
    "If you're unsure of something, it's okay to say so. "
    "Your goal is to make learning fun and informative!"
)

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Groq-based text generation API"}

@app.post("/generate-text")
async def generate_text(input: TextInput) -> TextOutput:
    try:
        client = get_groq_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": input.text}]},
        ]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
        )

        if not response.choices:
            raise ValueError("No response received from the model")

        return TextOutput(response=response.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") from e

