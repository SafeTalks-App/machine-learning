import discord
from discord.ext import commands
import tensorflow as tf
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv
import os

# Load file .env
load_dotenv()

# Ambil variabel dari environment
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Model configuration
vocab_size = 20000
max_length = 30
oov_token = '<OOV>'

# Emoji preprocessing
emoji_dict = {
    r'[:\)\(]+': 'emoji_positive',
    r'[:\+\-\|\/]+': 'emoji_neutral',
    r'[:\(]+': 'emoji_negative',
    r'[😊👍🌟😎]': 'emoji_positive',
    r'[😢😣😠😤]': 'emoji_negative',
    r'[😏🤔🤷]': 'emoji_neutral'
}

def preprocess_emojis(text):
    for pattern, replacement in emoji_dict.items():
        text = re.sub(pattern, f' {replacement} ', text)
    return text

# Load tokenizer
with open('model/lstm/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load TFLite model (LSTM only)
interpreter = tf.lite.Interpreter(model_path='model/lstm/lstm_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = preprocess_emojis(text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    return padded

def predict_text(text):
    padded = preprocess_text(text)
    padded = padded.astype(np.float32)  # Gunakan float32 jika model TFLite dinamis

    interpreter.set_tensor(input_details[0]['index'], padded)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_class = np.argmax(output, axis=1)[0]
    confidence = output[0][pred_class]
    class_names = {0: 'Hate Speech', 1: 'Offensive', 2: 'Neither'}
    return class_names[pred_class], confidence, output[0]

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    text = message.content
    pred_class, confidence, scores = predict_text(text)

    # Cari channel bernama "general"
    mod_channel = discord.utils.get(message.guild.text_channels, name="general")

    # Kalau nggak ada, pakai channel pertama yang bisa dikirim pesan
    if mod_channel is None:
        for channel in message.guild.text_channels:
            if channel.permissions_for(message.guild.me).send_messages:
                mod_channel = channel
                break

    if mod_channel is None:
        print("❌ ERROR: Tidak ada channel yang bisa dikirim pesan.")
        return

    # Moderation logic
    if pred_class == 'Hate Speech' and confidence > 0.85:
        await message.delete()
        await message.channel.send(
            f'{message.author.mention}, your message was removed due to hate speech (confidence: {confidence:.2f}).'
        )
        await mod_channel.send(
            f'Hate Speech detected: "{text}" by {message.author} (confidence: {confidence:.2f})'
        )
    elif pred_class == 'Offensive' and confidence > 0.85:
        await message.delete()
        await message.channel.send(
            f'{message.author.mention}, your message was removed due to offensive language (confidence: {confidence:.2f}).'
        )
        await mod_channel.send(
            f'Offensive detected: "{text}" by {message.author} (confidence: {confidence:.2f})'
        )
    elif confidence < 0.85:
        await mod_channel.send(
            f'Low-confidence prediction: "{text}" by {message.author} '
            f'(class: {pred_class}, confidence: {confidence:.2f}, scores: {scores})'
        )

    await bot.process_commands(message)

# Run bot
bot.run(BOT_TOKEN)
