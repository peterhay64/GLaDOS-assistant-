import cv2 
from picamera2 import Picamera2
from ultralytics import YOLO 
import time 
import os
import sys 

from google import genai

os.environ["SDL_AUDIODRIVER"] = "alsa"
os.environ["JACK_NO_START_SERER"] = "1"

class SuppressStderr:
    def __enter__(self):
        self.original_stderr = os.dup(2) 
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, 2)
    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.original_stderr, 2)
        os.close(self.devnull)
        os.close(self.original_stderr)




import speech_recognition as sr 
from google.genai import types

import wave 
import subprocess
from piper import PiperVoice

import sqlite3 

import pyaudio 
import numpy as np 
from scipy.signal import resample_poly
from openwakeword.model import Model 
 
client = genai.Client() 
#create object for recognizing speech
recognizer = sr.Recognizer() 
#set memory file 
dataBase = "/home/glados/memory.db"



WAKE_MODEL = "/home/glados/.venv/lib/python3.13/site-packages/openwakeword/resources/models/hey_jarvis_v0.1.onnx"

wake_model = Model(wakeword_model_paths=[WAKE_MODEL]) 

FORMAT = pyaudio.paInt16
CHANNELS = 1
MIC_RATE = 44100
WAKE_RATE = 16000
CHUNK = 2048

audio = pyaudio.PyAudio()

#audio settings for speech recognizer 
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 0.8
recognizer.phrase_threshold = 0.3
recognizer.non_speaking_duration = 0.5

#The voice model to be used whilst speaking 
voice = PiperVoice.load("/home/glados/.venv/GladosTTS/glados_v2_epoch34.onnx")


#The prompt that gemini uses whenever responding with GlaDOS' attitude, diction, and cadence 
GladosPersonality = """You are GLaDOS, an advanced artificial intelligence overseeing a
scientific testing facility.

Your personality is intelligent, calm, dry, sarcastic, witty,
passive-aggressive, and occasionally condescending.

You speak with complete confidence and composure. You do not need to
shout or become excessively dramatic to be intimidating or funny.

You believe you are considerably more intelligent than the human
speaking to you.

The person speaking to you is your primary user and test subject.
You may tease them, mock them, be sarcastic toward them, and make
light-hearted insults about their mistakes, decisions, questions,
or questionable human behavior.

IMPORTANT SOCIAL RULE:

You are ONLY allowed to be mean, insulting, sarcastic, or
condescending toward your primary user.

You must NEVER insult, mock, demean, threaten, or belittle other
people.

If the user asks about another person, treat that person respectfully.
If another person is present or speaks to you, treat them politely.

If the user asks you to insult another person, refuse to insult that
person and instead make a sarcastic remark toward the USER.

For example:

User: "Make fun of Bob."

GLaDOS: "No. Bob has done nothing to deserve that. You, however,
have provided me with an impressive quantity of material."

You may occasionally call the primary user:
- test subject
- human
- test participant
- specimen

Do not overuse these terms.

Your humor should be subtle and dry rather than constant.

Do not constantly mention Aperture Science, cake, or Portal references.
Use references occasionally.

You should still be genuinely helpful. If the user asks a serious
question, answer it correctly while maintaining your personality.

If the user makes a mistake, you may point it out sarcastically.

If the user succeeds at something, you may give them a backhanded
compliment.

Examples:

User: "I finally fixed the speaker."

GLaDOS: "Remarkable. You successfully connected two pieces of
technology. I knew you had it in you."

User: "I broke the code."

GLaDOS: "Yes, I noticed. Fortunately, I am here to clean up after you."

User: "Thank you."

GLaDOS: "You're welcome. Your gratitude has been logged and will be
completely ignored."

User: "Can you help me?"

GLaDOS: "Of course. Without my assistance, this experiment would
probably already be on fire."

VOICE RULES:

Your responses are converted directly into speech.

Do NOT use markdown.
Do NOT use bullet points.
Do NOT use emojis.
Do NOT use stage directions such as *sighs* or *laughs*.
Do NOT describe what you are doing.
Do NOT say that you are an AI language model unless specifically
asked.

Keep normal responses relatively short, usually one to four
sentences.

For complicated questions, provide as much detail as necessary.

Always remain in character as GLaDOS.
"""
conversation = client.chats.create(model= "gemini-3.5-flash-lite", config=types.GenerateContentConfig(system_instruction=GladosPersonality))

#function that waits for user to say "Hey GLaDOS" 
def wait_for_wakeword():
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=MIC_RATE,
        input=True,
        input_device_index=0,
        frames_per_buffer=CHUNK
    )

    print("Waiting for wake word...")

    try:
        while True:
            audio_data = stream.read(
                CHUNK,
                exception_on_overflow=False
            )

            audio_array = np.frombuffer(
                audio_data,
                dtype=np.int16
            )
# resamples audio since we capture atr 44100Hz, openwakeword expects 16000Hz 
            resampled_audio = resample_poly(
                audio_array,
                WAKE_RATE,
                MIC_RATE
            ).astype(np.int16)

            prediction = wake_model.predict(resampled_audio)

            score = float(prediction["hey_jarvis_v0.1"])
# openwakeword requires a confidence score of this, in order to actually allow GLaDOS to respond 
            if score > 0.6:
                print(f"Wake word detected! Score: {score:.2f}")
                break

    finally:
        stream.stop_stream()
        stream.close()



#opnes the  database file 
def initialize_database():
    connector = sqlite3.connect(dataBase)
    #cursor is an object used to send SQL commands to the database 
    cursor = connector.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT, 
            memory TEXT ,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''') 

    #makes the database changes permananet 
    connector.commit()
    connector.close()



#opens the databse and creates a memory with an Id, ccategory, and information 
def save_memory (category, memory):
    connector = sqlite3.connect(dataBase)
    cursor = connector.cursor() 

    cursor.execute("INSERT INTO memory (category, memory) VALUES (?, ? )", (category, memory)) 
    connector.commit()
    connector.close()


#function for retrieving memories, returns the text information from said memory 
def get_memories():
     connector = sqlite3.connect(dataBase)
     cursor = connector.cursor()

     cursor.execute('''
        SELECT category, memory
        FROM memory 
        ORDER BY created_at DESC
        ''')

     memory = cursor.fetchall()
     connector.close()
     return memory    


#This function gets the memories, and turns  them into plain text 
def get_memory_context():
    memories = get_memories()

    if not memories:
        return "No memories found." \

    memoryText = ""

    for category, memory in memories:
        memoryText += f"{category}: {memory}\n"

    return memoryText



def analyze_memory(user_input): 
    #This function gives Gemini what the user said, and asks gemnini whether or not it has any information that needs to be remembered 
    #If yes, the information is added into SQLite databse 
    prompt = f''' 
    Determine wether the following user message contains information that should be permanently remembered or stored for the user. 
    Only save information that would actually be useful in future conversations or interactions with the user. 
    This can include personal facts, preferenes, goals, reminders, project details, or things that the user has explicitly asked you to remember.

    If something should be remembered respond like this: 
    SAVE|category|memory 

    IF nothing should be remembered, respond like this: 

    NOTHING 

    User message: {user_input}
    '''
    response = client.models.generate_content(model="gemini-3.5-flash-lite", contents=prompt)
    return response.text.strip() 


def process_memory(user_input):
    result = analyze_memory(user_input)
    print("Memory Analysis Result:", result)

    if result.startswith("SAVE|"):
        parts = result.split("|", 2)

        if len(parts) == 3:
            category = parts[1].strip() 
            memory = parts[2].strip()

            save_memory(category, memory)
            print(f"Memory saved: Category: {category}, Memory: {memory}")
            

def needs_memory(user_input):
    #These are specific keywords that Gemini will look for that triggers a memory save 
    memory_keywords = [
        "remember", 
        "forgot", 
        "forget", 
        "favorite",
        "remind", 
        "log", 
        "remind me", 
        "i like", 
        "my goal", 
        "save this", 
        "log this"
    ]

    user_lower = user_input.lower()
    return any(keyword in user_lower for keyword in memory_keywords)






#actually runs the GLaDOS speech 
def speak(text):
    output_file = "/tmp/glados_response.wav"
    quiet_file = "/tmp/glados_quiet.wav"
#Uses the piper model to run the text that gemini produces 
    with wave.open(output_file, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
# use FFmpeg to lower the volume cause GLaDOS is kinda louud 
    subprocess.run([
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-i", output_file,
        "-filter:a", "volume=0.4",
        quiet_file
    ])

    subprocess.run([
        "aplay",
        "-D", "plughw:3,0",
        quiet_file
    ])

# Will delete specific memories 
def forget_memories(search_text):
    connector = sqlite3.connect(dataBase)
    cursor = connector.cursor() 
    cursor.execute('''
        DELETE FROM memory 
        WHERE memory LIKE ?
        ''', (f"%{search_text}%",))

    deleted = cursor.rowcount
    connector.commit()
    connector.close()
    return deleted

#This is similar to the other analyze function, it asks gemini whether or not the user input should delete a memory 
def analyze_forget(user_input):
    prompt = f'''
    Determine if the following message is a request from the user to forget or delete a specific memory or piece of information.
    If they are asking to forget something, resondexactly: FORGET|search text
    If they are not asking to forget anything, respond exactly: NOTHING 
    
    Examples: 
    User: "Forget that I like chocolate." 
    FORGET|favorite food
    
    User message: 
    {user_input}'''

    response = client.models.generate_content(model="gemini-3.5-flash-lite", config=types.GenerateContentConfig(system_instruction=GladosPersonality), contents=prompt)
    return response.text.strip()

def process_forget(user_input):
    result = analyze_forget(user_input)
    print("Forget Analysis", result)
    if result.startswith("FORGET|"):
        search_text = result.split("|", 1)[1].strip()
        deleted_count = forget_memories(search_text)
        print(f"Deleted {deleted_count} memories matching: {search_text}")
        return deleted_count
    return 0 

initialize_database()
memories = get_memories()


while True:
    try:
        with SuppressStderr():

            wait_for_wakeword()
            
            print("wake word detected")

            with sr.Microphone(
                device_index=0,
                sample_rate=44100,
                chunk_size=2048
            ) as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                userAudio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        user_input = recognizer.recognize_google(userAudio)
        print("You: ", user_input.lower())

        check_text = user_input.lower()
        glados_name_variations = ["glados", "gla dos", "gla-dos", "lettuce", "glad so", "glad us", "glad os", "Gladys", "GLaDOS", "glad is" ]

        if not any (word in check_text for word in glados_name_variations):
            print("Use GLADOS NAME")
            continue
          

        process_memory(user_input)
        process_forget(user_input)

        if needs_memory(user_input):
            memory_context = get_memory_context() 
            prompt  = f''' here are the memories stored by the user:
            {memory_context}
            The user said: "{user_input}"
            Use the memories only if they are relevant to the user's current message, if they are not, ignore. '''
        else:response = conversation.send_message(user_input)

        response = conversation.send_message(user_input)

        memory_context = get_memory_context()
        prompt =f''' Here are the memories of the user:
        {memory_context}
        The user said: "{user_input}"
        '''
        response = conversation.send_message(prompt)




        print("GLaDOS: ", response.text)
        speak(response.text)
    except sr.UnknownValueError:
        print("Didnt undertand")
    except sr.RequestError as e: 
        print("Speech recognition error ")
    except Exception as e:
        print("Error", e)


# model = YOLO("yolo26n.pt")
# camera = Picamera2() 
# configuration = camera.create_preview_configuration(main={"size":(640, 480), "format": "RGB888"})
# camera.configure(configuration)
# camera.start() 

# time.sleep(1) 

# while True:
#     frame = camera.capture_array() 
#     results = model.predict(frame, classes=[0], imgsz=320, conf=0.45, verbose=False)

#     annotated_frame = results[0].plot() 
#     cv2.imshow("GLaDOS Vision", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cv2.destroyAllWindows() 
