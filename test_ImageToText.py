from groq import Groq
import os
import base64
import requests
from typing import Union
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")

#GROQ_API_KEY = "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"  # Replace with your actual API key

client = Groq(api_key=GROQ_API_KEY)


def analyze_image(image_data: Union[str, bytes], prompt: str = "") -> str:
    """Analyze an image using Groq's vision capabilities."""
    try:
        image_bytes = None

        if isinstance(image_data, str):
            if image_data.startswith("http://") or image_data.startswith("https://"):
                response = requests.get(image_data)
                if response.status_code == 200:
                    image_bytes = response.content
                else:
                    raise ValueError(f"Failed to fetch image from URL: {image_data}")
            elif os.path.exists(image_data):
                with open(image_data, "rb") as f:
                    image_bytes = f.read()
            else:
                raise ValueError("Invalid image path or URL.")
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            raise ValueError("image_data must be a file path, URL, or bytes")

        if not prompt:
            prompt = "Please describe what you see in this image in detail."

        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            max_tokens=1000,
        )

        if not response.choices:
            raise ValueError("No response received from the vision model")

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"Failed to analyze image: {e}")


# Example usage
image = "./Images/football.webp"
result = analyze_image(image)
print("Image Analysis Result: \n", result)
