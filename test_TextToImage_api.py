import os
from together import Together
import base64

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise RuntimeError("Missing TOGETHER_API_KEY in environment.")


client = Together(api_key=TOGETHER_API_KEY)

prompt = "Create a detailed image of a futuristic city skyline at sunset, with flying cars and neon lights. The scene should be vibrant and full of life, showcasing advanced technology and architecture. Include elements like holographic billboards and lush greenery integrated into the urban landscape."


response = client.images.generate(
                prompt=prompt,
                model="black-forest-labs/FLUX.1-schnell-Free",
                width=1024,
                height=768,
                steps=4,
                n=1,
                response_format="b64_json",
        )

image_data = base64.b64decode(response.data[0].b64_json)

output_path = "./Images/futuristic_city_skyline.png"
with open(output_path, "wb") as f:
    f.write(image_data)
print(f"Image saved to {output_path}")

