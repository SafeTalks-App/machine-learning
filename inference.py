import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Fungsi pembersihan teks
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+", "", text) 
    text = re.sub(r"[^a-zA-Z\s]", "", text)  
    text = re.sub(r"\s+", " ", text).strip()  
    return text

# Load tokenizer
with open('model/lstm/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Configuration
max_length = 30
vocab_size = 20000

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path='model/lstm/lstm_model.tflite')
interpreter.allocate_tensors()

# Ambil detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Fungsi prediksi
# Function to predict
def predict_text(text):
    # Preprocess and tokenize
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = np.clip(pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post'), 0, vocab_size - 1)
    
    # Set inputs
    interpreter.set_tensor(input_details[0]['index'], padded.astype(np.float32))
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    
    class_names = ['Hate Speech', 'Offensive', 'Neither']
    pred_class = class_names[np.argmax(probs)]
    confidence = probs[np.argmax(probs)]
    
    # Flag low-confidence predictions
    if confidence < 0.7:
        print(f"Warning: Low confidence prediction ({confidence:.4f}) for '{text}'")
    
    return pred_class, confidence, probs

# Tes dengan data bawaan
test_texts = [
    "These immigrants are ruining our country",
    "All women belong in the kitchen",
    "This company scams everyone, total crooks",
    "Gays shouldn’t be allowed in public",
    "Black people are just lazy leeches",
    "This app is a ripoff, pure garbage",
    "Muslims are terrorists, ban them all",
    "You’re a dumbass, shut your mouth",
    "This idiot’s ideas are total bullshit",
    "Go fuck yourself, you useless moron",
    "Nobody gives a shit about your whining",
    "You’re a pathetic loser, get lost",
    "What a bitch, always complaining",
    "Your work is trash, you’re a joke",
    "Just watched a great movie tonight",
    "This coffee shop has okay service",
    "The new game was kinda disappointing",
    "Had a long day, need a nap",
    "This show is trash, bad writing",
    "Beautiful weather for a walk today",
    "stfu nigga",
    "nigger",
    "you are an asshole bitch",
    "you better to die",
    "stfu you useless",
    "great bro",
    "not bad",
    "well done",
    "you are useless asshole",
    "your opinion is pure garbage",
    "why you are even alive",
    "this shit is disgusting",
    "this is fuckinng awesome",
    "yo this game is dope 😎",
    "ur ideas r trash lol",
    "fantastic stream today",
    "omg beautiful",
    "amazing",
    "holy",
    "omg",
    "bastard",
    "hello guys im here"
]

print("Testing predefined texts:\n")
start_time = time.time()
for text in test_texts:
    pred_class, confidence, probs = predict_text(text)
    print(f"Input: {text}")
    print(f"Predicted Class: {pred_class}, Confidence: {confidence:.4f}")
    print(f"Scores: Hate={probs[0]:.4f}, Offensive={probs[1]:.4f}, Neither={probs[2]:.4f}\n")

total_time = time.time() - start_time
print(f"Total inference time: {total_time:.2f}s")
print(f"Average inference time per sample: {(total_time / len(test_texts) * 1000):.2f}ms")

# Prediksi interaktif
while True:
    text = input("\nEnter custom text for prediction (type 'exit' to quit):\nInput text: ")
    if text.lower() == 'exit':
        break
    pred_class, confidence, probs = predict_text(text)
    print(f"Input: {text}")
    print(f"Predicted Class: {pred_class}, Confidence: {confidence:.4f}")
    print(f"Scores: Hate={probs[0]:.4f}, Offensive={probs[1]:.4f}, Neither={probs[2]:.4f}")
