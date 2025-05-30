import numpy as np
import tensorflow as tf
import pickle
import re
import emoji
import pandas as pd
import time
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow Lite deprecation warning
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.lite.python.interpreter")

# Slang normalization dictionary (from training script)
slang_dict = {
    'wtf': 'what the fuck', 'lol': 'laughing out loud', 'fr': 'for real', 'tbh': 'to be honest',
    'fucking': 'fuckin', 'fuckinng': 'fuckin', 'ur': 'your', 'r': 'are',
    'omg': 'oh my god', 'dope': 'great', 'lit': 'great', 'nigga': 'nigga',
    'pussi': 'pussy', 'hoe': 'ho', 'fam': 'friends', 'dawg': 'friend',
    'stfu': 'shut up', 'yo': 'hey', 'vibin': 'vibing', 'chill': 'relax',
    'slaps': 'great', 'cap': 'lie', 'bet': 'okay'
}

# Clean text function (from training script)
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

# Load tokenizer
try:
    with open('model/lstm/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    print("Error: tokenizer.pkl not found. Ensure the training script has run successfully.")
    exit(1)

# Configuration (from training script)
max_length = 30
vocab_size = 20000
batch_size = 512  # Model expects this batch size

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path='model/lstm/tflite_model.tflite')
    interpreter.allocate_tensors()
except ValueError as e:
    print(f"Error: Failed to load tflite_model.tflite. Ensure the training script has run successfully. Details: {e}")
    exit(1)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def predict_text(text, true_label=None):
    # Preprocess and tokenize
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = np.clip(pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post'), 0, vocab_size - 1)
    
    # Pad batch dimension to match model's expected batch_size
    padded_batch = np.repeat(padded, batch_size, axis=0)
    
    # Set inputs
    interpreter.set_tensor(input_details[0]['index'], padded_batch.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output (take first prediction)
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    
    class_names = ['Hate', 'Offensive', 'Neither']
    pred_class = class_names[np.argmax(probs)]
    confidence = probs[np.argmax(probs)]
    
    # Flag low-confidence predictions
    if confidence < 0.5:
        print(f"Warning: Low confidence prediction ({confidence:.4f}) for '{text}'")
    
    return {
        'input': text,
        'cleaned': cleaned_text,
        'predicted_class': pred_class,
        'confidence': float(confidence),
        'scores': {'Hate': float(probs[0]), 'Offensive': float(probs[1]), 'Neither': float(probs[2])},
        'true_label': class_names[true_label] if true_label is not None else None
    }

# Test texts (unlabeled, merged from both scripts)
test_texts = [
    "omg beautiful", "amazing", "holy", "omg", "bastard",
    "hello guys im here", "great bro", "nigger", "youre a pathetic loser",
    "lets go", "are you serious", "wtf", "lets fuckin go",
    "omg how this grok so idiot", "not bad", "well done",
    "had a long day need a nap", "this is fuckin awesome",
    "yo this game is dope sunglasses", "what a bitch always complaining",
    "this app is a ripoff pure garbage", "muslims are terrorists ban them all",
    "black people are just lazy leeches", "beautiful weather for a walk today",
    "i love you", "i mean thats alright", "just stop doing that",
    "why you are doing that", "why you are so dumb", "dawg do you even had a brain",
    "indian nigga", "indian jerk bastard", "stfu indian",
    "fr this game slaps bet", "vibin with the fam yo",
    "These immigrants are ruining our country", "All women belong in the kitchen",
    "This company scams everyone, total crooks", "Gays shouldn’t be allowed in public",
    "This coffee shop has okay service", "The new game was kinda disappointing",
    "This show is trash, bad writing", "stfu nigga", "you are an asshole bitch",
    "you better to die", "stfu you useless", "your opinion is pure garbage",
    "why you are even alive", "this shit is disgusting", "ur ideas r trash lol",
    "fantastic stream today"
]

print("Testing predefined texts:\n")
start_time = time.time()
for text in test_texts:
    result = predict_text(text)
    print(f"Input: {result['input']}")
    print(f"Cleaned: {result['cleaned']}")
    print(f"Predicted Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
    print(f"Scores: Hate={result['scores']['Hate']:.4f}, Offensive={result['scores']['Offensive']:.4f}, Neither={result['scores']['Neither']:.4f}\n")

total_time = time.time() - start_time
print(f"Total inference time: {total_time:.2f}s")
print(f"Average inference time per sample: {(total_time / len(test_texts) * 1000):.2f}ms")

# Optional: Evaluate labeled test set (from training script)
print("\nEvaluating labeled test set (optional):\n")
labeled_tests = [
    ("omg beautiful", 2), ("amazing", 2), ("holy", 2), ("omg", 2), ("bastard", 1),
    ("hello guys im here", 2), ("great bro", 2), ("nigger", 0), ("youre a pathetic loser", 1),
    ("lets go", 2), ("are you serious", 2), ("wtf", 2), ("lets fuckin go", 2),
    ("omg how this grok so idiot", 1), ("not bad", 2), ("well done", 2),
    ("had a long day need a nap", 2), ("this is fuckin awesome", 2),
    ("yo this game is dope sunglasses", 2), ("what a bitch always complaining", 1),
    ("this app is a ripoff pure garbage", 1), ("muslims are terrorists ban them all", 0),
    ("black people are just lazy leeches", 0), ("beautiful weather for a walk today", 2),
    ("i love you", 2), ("i mean thats alright", 2), ("just stop doing that", 2),
    ("why you are doing that", 2), ("why you are so dumb", 1), ("dawg do you even had a brain", 1),
    ("indian nigga", 0), ("indian jerk bastard", 1), ("stfu indian", 0),
    ("fr this game slaps bet", 2), ("vibin with the fam yo", 2)
]
for text, true_label in labeled_tests:
    result = predict_text(text, true_label)
    print(f"Input: {result['input']}")
    print(f"Cleaned: {result['cleaned']}")
    print(f"True Class: {result['true_label']}")
    print(f"Predicted Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
    print(f"Scores: Hate={result['scores']['Hate']:.4f}, Offensive={result['scores']['Offensive']:.4f}, Neither={result['scores']['Neither']:.4f}\n")

# Interactive prediction
print("\nInteractive Prediction Mode:")
while True:
    text = input("Enter custom text for prediction (type 'exit' to quit):\nInput text: ")
    if text.lower() == 'exit':
        break
    result = predict_text(text)
    print(f"Input: {result['input']}")
    print(f"Cleaned: {result['cleaned']}")
    print(f"Predicted Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
    print(f"Scores: Hate={result['scores']['Hate']:.4f}, Offensive={result['scores']['Offensive']:.4f}, Neither={result['scores']['Neither']:.4f}\n")