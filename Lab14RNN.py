import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LambdaCallback


with open('Paul_Graham_Essays.txt', 'r', encoding='ISO-8859-1') as file:
    text = file.read().lower()

unique_chars = sorted(list(set(text)))
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for char, idx in char_to_index.items()}
vocab_size = len(unique_chars)


seq_length = 40
step = 3
inputs = []
targets = []

for i in range(0, len(text) - seq_length, step):
    seq = text[i: i + seq_length]
    next_char = text[i + seq_length]
    inputs.append([char_to_index[char] for char in seq])
    targets.append(char_to_index[next_char])

X = np.array(inputs)
y = to_categorical(targets, num_classes=vocab_size)


model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=seq_length),
    SimpleRNN(128),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(X, y, batch_size=128, epochs=20, verbose=0)


def sample(predictions, temperature=0.7):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-8) / temperature
    predictions = np.exp(predictions)
    predictions = predictions / np.sum(predictions)
    return np.argmax(np.random.multinomial(1, predictions, 1))


start = random.randint(0, len(text) - seq_length - 1)
seed_text = text[start: start + seq_length]
generated_text = seed_text

print(f"\nGenerating text with seed:\n\"{seed_text}\"\n")

for _ in range(600):
    input_seq = [char_to_index[char] for char in seed_text[-seq_length:]]
    input_seq = np.reshape(input_seq, (1, seq_length))
    prediction = model.predict(input_seq, verbose=0)[0]
    next_char_idx = sample(prediction, temperature=0.7)
    next_char = index_to_char[next_char_idx]
    
    generated_text += next_char
    seed_text += next_char

print(generated_text)
