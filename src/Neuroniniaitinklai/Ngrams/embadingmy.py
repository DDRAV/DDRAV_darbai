import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download

# Atsisiųskite reikalingus NLTK duomenis
download('punkt')
download('stopwords')

# Funkcija tekstams valyti
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Pašalinkite skyrybos ženklus
    text = re.sub(r'\d+', '', text)  # Pašalinkite skaičius
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Funkcija n-gramų generavimui
def generate_ngrams(text, n):
    tokens = text.split()
    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return ngrams

# Gaukite duomenis
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X_raw, y = newsgroups.data, newsgroups.target

# Išvalykite tekstus
X_cleaned = [clean_text(text) for text in X_raw]

# Sugeneruokite n-gramas iš anksto
def preprocess_ngrams(X_cleaned, n):
    return [" ".join(generate_ngrams(text, n)) for text in X_cleaned]

# Eksperimentų konfigūracija
n_values = [50,100]  # N-gramų reikšmės
max_words_values = [5000, 10000]  # Žodžių skaičius

results = {}

for n in n_values:
    # Sugeneruokite n-gramas vieną kartą
    X_ngrams = preprocess_ngrams(X_cleaned, n)
    X_train, X_test, y_train, y_test = train_test_split(X_ngrams, y, test_size=0.2, random_state=42)

    for max_words in max_words_values:
        # Teksto tokenizacija
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        max_sequence_length = 100
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length, padding='post')
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length, padding='post')

        # Modelio sukūrimas
        model = Sequential([
            Embedding(input_dim=max_words, output_dim=64, input_length=max_sequence_length),
            SimpleRNN(32, activation='relu'),
            Dense(len(np.unique(y)), activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Modelio apmokymas
        print(f"Apmokoma su n={n}, max_words={max_words}...")
        history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

        # Modelio testavimas
        loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=1)
        print(f"n={n}, max_words={max_words}, Accuracy={accuracy:.4f}")

        # Rezultatų saugojimas
        results[(n, max_words)] = accuracy

# Rezultatų atvaizdavimas grafike
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
for n in n_values:
    accuracies = [results[(n, mw)] for mw in max_words_values]
    ax.plot(max_words_values, accuracies, label=f'N-gram size: {n}', marker='o')

ax.set_title('Model Accuracy for Different N-Grams and Max Words')
ax.set_xlabel('Max Words')
ax.set_ylabel('Accuracy')
ax.legend()
plt.grid()
plt.show()
