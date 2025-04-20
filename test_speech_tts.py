import os
from elevenlabs import ElevenLabs, Voice, VoiceSettings
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in environment.")


text = "Hello, this is a test of the ElevenLabs text-to-speech API."

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

if not text.strip():
    raise ValueError("Input text cannot be empty")

if len(text) > 5000:  # ElevenLabs typical limit
    raise ValueError("Input text exceeds maximum length of 5000 characters")

try:
    audio_generator = client.generate(
        text=text,
        voice=Voice(
            voice_id="EXAVITQu4vr4xnSDxMaL",  # Replace with your actual voice ID
            settings=VoiceSettings(stability=0.5, similarity_boost=0.5),
        ),
        model="eleven_flash_v2_5",
    )

    # Convert generator to bytes
    audio_bytes = b"".join(audio_generator)
    if not audio_bytes:
        raise RuntimeError("Generated audio is empty")
    # Save to file
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
except Exception as e:
    raise RuntimeError(f"Text-to-speech conversion failed: {str(e)}") from e