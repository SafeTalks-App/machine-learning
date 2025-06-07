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

# Slang normalization dictionary
slang_dict = {
    'wtf': 'what the fuck', 'lol': 'laughing out loud', 'fr': 'for real', 'tbh': 'to be honest',
    'fucking': 'fuckin', 'fuckinng': 'fuckin', 'ur': 'your', 'r': 'are',
    'omg': 'oh my god', 'dope': 'great', 'lit': 'great', 'nigga': 'nigga',
    'pussi': 'pussy', 'hoe': 'ho', 'fam': 'friends', 'dawg': 'friend',
    'stfu': 'shut up', 'yo': 'hey', 'vibin': 'vibing', 'chill': 'relax',
    'slaps': 'great', 'cap': 'lie', 'bet': 'okay'
}

# Clean text function
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

# Configuration
max_length = 30
vocab_size = 20000

# Load TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path='model/lstm/lstm_model.tflite')
    interpreter.allocate_tensors()
except ValueError as e:
    print(f"Error: Failed to load lstm_model.tflite. Ensure the training script has run successfully. Details: {e}")
    exit(1)

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def predict_text(text, true_label=None):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = np.clip(pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post'), 0, vocab_size - 1)
    interpreter.set_tensor(input_details[0]['index'], padded.astype(np.float32))
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    class_names = ['Hate Speech', 'Offensive', 'Neither']
    pred_class = class_names[np.argmax(probs)]
    confidence = probs[np.argmax(probs)]
    if confidence < 0.5:
        print(f"Warning: Low confidence prediction ({confidence:.4f}) for '{text}'")
    return {
        'input': text,
        'cleaned': cleaned_text,
        'predicted_class': pred_class,
        'predicted_index': np.argmax(probs),
        'confidence': float(confidence),
        'scores': {'Hate Speech': float(probs[0]), 'Offensive': float(probs[1]), 'Neither': float(probs[2])},
        'true_label': class_names[true_label] if true_label is not None else None,
        'true_index': true_label
    }

# Labeled test set (merged and deduplicated with true labels: 0=Hate Speech, 1=Offensive, 2=Neither)
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
    ("fr this game slaps bet", 2), ("vibin with the fam yo", 2),
    ("These immigrants are ruining our country", 0), ("All women belong in the kitchen", 0),
    ("This company scams everyone, total crooks", 1), ("Gays shouldn’t be allowed in public", 0),
    ("This coffee shop has okay service", 2), ("The new game was kinda disappointing", 2),
    ("This show is trash, bad writing", 1), ("stfu nigga", 0), ("you are an asshole bitch", 1),
    ("you better to die", 1), ("stfu you useless", 1), ("your opinion is pure garbage", 1),
    ("why you are even alive", 1), ("this shit is disgusting", 1), ("ur ideas r trash lol", 1),
    ("fantastic stream today", 2)
]

# Evaluate labeled test set and compute accuracy
print("Evaluating labeled test set:\n")
start_time = time.time()
correct_predictions = 0
for text, true_label in labeled_tests:
    result = predict_text(text, true_label)
    print(f"Input: {result['input']}")
    print(f"Cleaned: {result['cleaned']}")
    print(f"True Class: {result['true_label']}")
    print(f"Predicted Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
    print(f"Scores: Hate Speech={result['scores']['Hate Speech']:.4f}, Offensive={result['scores']['Offensive']:.4f}, Neither={result['scores']['Neither']:.4f}")
    if result['predicted_index'] == result['true_index']:
        correct_predictions += 1
    print(f"Correct: {result['predicted_index'] == result['true_index']}\n")

accuracy = correct_predictions / len(labeled_tests) * 100
total_time = time.time() - start_time
print(f"Total inference time: {total_time:.2f}s")
print(f"Average inference time per sample: {(total_time / len(labeled_tests) * 1000):.2f}ms")
print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(labeled_tests)} correct)")

# Interactive prediction mode
print("\nInteractive Prediction Mode:")
while True:
    text = input("Enter custom text for prediction (type 'exit' to quit):\nInput text: ")
    if text.lower() == 'exit':
        break
    result = predict_text(text)
    print(f"Input: {result['input']}")
    print(f"Cleaned: {result['cleaned']}")
    print(f"Predicted Class: {result['predicted_class']}, Confidence: {result['confidence']:.4f}")
    print(f"Scores: Hate Speech={result['scores']['Hate Speech']:.4f}, Offensive={result['scores']['Offensive']:.4f}, Neither={result['scores']['Neither']:.4f}\n")