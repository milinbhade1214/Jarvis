import os
from groq import Groq
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")

client = Groq(api_key=GROQ_API_KEY)

# System prompt to set tone
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are Jarvis, a witty and helpful assistant. "
        "Keep things friendly, engaging, and safe for all users. "
        "Avoid any harmful or offensive language. Be clear and helpful."
    ),
}


def chat_with_groq(message, history):
    """Chat handler for Gradio interface."""
    messages = [SYSTEM_PROMPT]

    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1000,
        temperature=0.7,
    )

    reply = response.choices[0].message.content
    return reply 


if __name__ == "__main__":
    gr.ChatInterface(fn=chat_with_groq, title="💬 Chat with Jarvis").launch()
