import discord
from discord.ext import commands
import tensorflow as tf
import numpy as np
import pickle
import re
import emoji
import pandas as pd
import os
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# ✅ REGISTER CUSTOM LAYER & LOSS
# =========================
@tf.keras.utils.register_keras_serializable()
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape=(input_shape[-1], 1),
                                      initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(name="bias", shape=(input_shape[1], 1),
                                    initializer="zeros", trainable=True)
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.tensordot(inputs, self.kernel, axes=1) + self.bias)
        alpha = tf.keras.backend.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * inputs, axis=1)
        return context

@tf.keras.utils.register_keras_serializable()
def focal_loss_fn(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_sum(loss, axis=1)

# =========================
# 🔤 Normalisasi Slang
# =========================
slang_dict = {
    'wtf': 'what the fuck', 'lol': 'laughing out loud', 'fr': 'for real', 'tbh': 'to be honest',
    'fucking': 'fuckin', 'fuckinng': 'fuckin', 'ur': 'your', 'r': 'are',
    'omg': 'oh my god', 'dope': 'great', 'lit': 'great', 'nigga': 'nigga',
    'pussi': 'pussy', 'hoe': 'ho', 'fam': 'friends', 'dawg': 'friend',
    'stfu': 'shut up', 'yo': 'hey', 'vibin': 'vibing', 'chill': 'relax',
    'slaps': 'great', 'cap': 'lie', 'bet': 'okay'
}

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    text = text.lower()
    text = emoji.demojize(text, delimiters=(' ', ' '))
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s!?]', '', text)
    for slang, full in slang_dict.items():
        text = re.sub(r'\b' + slang + r'\b', full, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# 🔃 Load Tokenizer & Model
# =========================
try:
    with open('model/lstm/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Error: tokenizer.pkl not found.")
    exit(1)

try:
    model = tf.keras.models.load_model('model/lstm/lstm_model.keras')
except Exception as e:
    print(f"Error loading keras model: {e}")
    exit(1)

# =========================
# ⚙️ Konfigurasi
# =========================
vocab_size = 20000
max_length = 30

# =========================
# 🤖 Discord Bot Setup
# =========================
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# =========================
# 🧠 Fungsi Prediksi
# =========================
def predict_text(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = np.clip(pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post'), 0, vocab_size - 1)
    
    probs = model.predict(padded, verbose=0)[0]
    class_names = ['Hate Speech', 'Offensive', 'Neither']
    pred_class = class_names[np.argmax(probs)]
    confidence = probs[np.argmax(probs)]

    return {
        'cleaned': cleaned_text,
        'predicted_class': pred_class,
        'confidence': float(confidence),
        'scores': {'Hate Speech': float(probs[0]), 'Offensive': float(probs[1]), 'Neither': float(probs[2])}
    }

# =========================
# 📩 Event Handling
# =========================
@bot.event
async def on_ready():
    print(f'Bot logged in as {bot.user}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    content = message.content

    # ✅ Command manual test
    if content.startswith("!test "):
        test_text = content[6:].strip()
        result = predict_text(test_text)

        await message.channel.send(
            f'📊 **Prediction Test**\n'
            f'**Original:** {test_text}\n'
            f'**Cleaned:** {result["cleaned"]}\n'
            f'**Class:** {result["predicted_class"]} ({result["confidence"]:.2f})\n'
            f'**Scores:**\n'
            f'   • Hate Speech: {result["scores"]["Hate Speech"]:.4f}\n'
            f'   • Offensive: {result["scores"]["Offensive"]:.4f}\n'
            f'   • Neither: {result["scores"]["Neither"]:.4f}'
        )
        return

    # 🔮 Auto prediction
    result = predict_text(content)
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]

    # ⚠️ Automated moderation
    if predicted_class == "Hate Speech" and confidence >= 0.80:
        await message.delete()
        await message.channel.send(
            f'🚫 Message from {message.author.mention} was removed due to **Hate Speech**.'
        )
    elif predicted_class == "Offensive" and confidence >= 0.85:
        await message.channel.send(
            f'⚠️ Warning to {message.author.mention}: your message contains **Offensive Content**.'
        )

    await bot.process_commands(message)


# =========================
# 🚀 Run Bot
# =========================
bot.run(BOT_TOKEN)
