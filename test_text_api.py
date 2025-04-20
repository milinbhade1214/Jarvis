from groq import Groq
import os
from typing import Union
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")

client = Groq(api_key=GROQ_API_KEY)

prompt = "Who is the current prime minister of the UK?"

system_prompt = (
    "You are an AI assistant with a witty and humorous personality. "
    "You're helpful, engaging, and intelligent—but always respectful. "
    "Use clever, light humor where appropriate, but never be sarcastic, offensive, or biased. "
    "Stay factual, avoid speculation, and respond in clear, friendly language. "
    "If you're unsure of something, it's okay to say so. "
    "Your goal is to make learning fun and informative!"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": [{"type": "text", "text": prompt}]},
]



response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
    )

if not response.choices:
    raise ValueError("No response received from the vision model")

print(response.choices[0].message.content)
