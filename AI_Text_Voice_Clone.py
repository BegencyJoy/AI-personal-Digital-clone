import os
import json
import random
import asyncio
import torch
import speech_recognition as sr
from datetime import datetime
from telethon import TelegramClient, events
from TTS.api import TTS
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# CONFIGURATION

api_id = 1234              
api_hash = "1234"
GENAI_API_KEY = "qwert"
PERSONALITY_FILE = r"D:\python2.0\AI-personal-DIgital_Twin\converted_dataset.jsonl"
VOICE_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
VOICE_SAMPLE = r"D:\python2.0\AI-personal-DIgital_Twin\myvoice_fixed.wav"
ALLOWED_SENDERS = [  "sathya_sjr", "valentine_joy","sachin_sur","sakthi_siv","achu_1803"]
LOG_DIR = "logs"


# INITIALIZATION

genai.configure(api_key=GENAI_API_KEY)
tts = TTS(model_name=VOICE_MODEL, progress_bar=False, gpu=torch.cuda.is_available())
client = TelegramClient("clone_session", api_id, api_hash)
model = SentenceTransformer("all-MiniLM-L6-v2")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)


# LOAD PERSONALITY DATASET

with open(PERSONALITY_FILE, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]

dataset_prompts = [s["prompt"] for s in dataset]
dataset_embeddings = model.encode(dataset_prompts, convert_to_tensor=True)

def get_relevant_example(user_input):
    input_emb = model.encode(user_input, convert_to_tensor=True)
    sim = util.cos_sim(input_emb, dataset_embeddings)
    best_idx = torch.argmax(sim).item()
    return dataset[best_idx]

def build_personality_prompt(example):
    return f"User: {example['prompt']}\nMe: {example['response']}\n\n"


# FIXED ANSWERS

fixed_answers = {
    "what is your name": "I am Begency."
}

# TEXT RESPONSE GENERATION

def generate_clone_response(user_input):
    normalized = user_input.strip().lower()
    if normalized in fixed_answers:
        return fixed_answers[normalized]

    example = get_relevant_example(user_input)
    few_shot = build_personality_prompt(example)

    prompt = f"""
You are Begency's digital twin.
Reply naturally and in Begency's tone."

Example of Begency's style:
{few_shot}

User: {user_input}
Me:
"""
    model_ai = genai.GenerativeModel("gemini-2.5-flash")
    response = model_ai.generate_content(prompt)
    return response.text.strip()

# VOICE GENERATION

def generate_voice_reply(text, output_path="clone_reply.wav"):
    tts.tts_to_file(text=text, file_path=output_path, speaker_wav=VOICE_SAMPLE, language="en")
    return output_path

# VOICE MESSAGE TRANSCRIPTION

def transcribe_voice(file_path):
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = r.record(source)
    return r.recognize_google(audio_data)


# LOGGING

def get_daily_log_file():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"log_{today}.txt")

def log_interaction(username, phone, user_message, reply_text, reply_type):
    log_file = get_daily_log_file()
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        log.write(f"\nFrom: {username or phone}")
        log.write(f"\nMessage: {user_message}")
        log.write(f"\nReply ({reply_type}): {reply_text}")
        log.write("\n" + "=" * 60 + "\n")

# TELEGRAM HANDLER

@client.on(events.NewMessage)
async def handler(event):
    sender = await event.get_sender()
    username = (sender.username or "").lower()
    phone = getattr(sender, "phone", None)
    today = datetime.now().date()
    msg_date = event.date.date()

    allowed_normalized = [x.lower() for x in ALLOWED_SENDERS]
    if username not in allowed_normalized and (not phone or f"+{phone}" not in allowed_normalized):
        print(f"Ignored message from: {username or phone}")
        return

    if msg_date != today:
        print(f"Ignored old message from {username or phone}")
        return

    message = event.raw_text.strip() if event.raw_text else ""

    # Handle Voice Message
    if event.message.voice:
        print(f"Voice message from {username or phone}")
        voice_path_ogg = await event.download_media(file="voice.ogg")
        wav_path = "voice_input.wav"
        AudioSegment.from_file(voice_path_ogg).export(wav_path, format="wav")

        try:
            user_text = transcribe_voice(wav_path)
            print(f"Transcribed: {user_text}")
            reply_text = generate_clone_response(user_text)
            reply_voice = generate_voice_reply(reply_text)
            await client.send_file(event.chat_id, reply_voice, voice_note=True)
            log_interaction(username, phone, user_text, reply_text, "voice")
            print("Voice reply sent.")
        except Exception as e:
            print(f"Voice processing error: {e}")
        return

    # Handle Text Message
    if message:
        print(f"Text from {username or phone}: {message}")
        reply_text = generate_clone_response(message)
        await event.respond(reply_text)
        log_interaction(username, phone, message, reply_text, "text")
        print("Text reply sent.")


# RUN BOT

print(" Begency Digital Twin v3 started. Listening for today's messages only...")
with client:
    client.loop.run_forever()
