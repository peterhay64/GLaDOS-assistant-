from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs 
from elevenlabs.play import play 
import base64
import os

elevenlabs = ElevenLabs( api_key=os.getenv("ELEVENLABS_API_KEY"),


)

voices = elevenlabs.text_to_voice.design(model_id = "eleven_multilingual_ttv_v2",
    voice_description="Female synthetic AI voice. Mature, articulate, highly controlled delivery. Netural American accent. Slightly low pitch. Extremley precise pronunciation, Clam and cynical, with a subtle  sense of superiority. Sepaks slowly and deliberately, often pausing before important words. EMotionally restrained, but capable of sarcasm. Clean studio recording with a substle futurisitc computer quality. Condescending to the user, but always helpful, similar to GLaDOS",
    text="Hello, and welcome to the labratory. My namde is GLADOS and I will be wathcing over your tests for the next few days. It is imparative that you follow all commands, and complete tests as fast as possible.")
    



for preview in voices.previews: 
    audio_buffer = base64.b64decode(preview.audio_base_64)

    print(f"Playing preview: {preview.generated_voice_id}")

    play(audio_buffer)
