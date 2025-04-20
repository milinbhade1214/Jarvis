import os
from elevenlabs import ElevenLabs, Voice, VoiceSettings
from groq import Groq
import tempfile
#                         model="whisper-1",
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")

# How to load the audio data
audio_data_path = './output.wav'
with open(audio_data_path, "rb") as audio_file:
    audio_data = audio_file.read()

client = Groq(api_key=GROQ_API_KEY)


if not audio_data:
    raise ValueError("Audio data cannot be empty")


try:
    # Create a temporary file with .wav extension
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(audio_data)
        temp_file_path = temp_file.name

    try:
        # Open the temporary file for the API request
        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
            )

        if not transcription:
            raise RuntimeError("Transcription result is empty")

        print("Transcription:", transcription)

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

except Exception as e:
    raise RuntimeError(f"Speech-to-text conversion failed: {str(e)}") from e


