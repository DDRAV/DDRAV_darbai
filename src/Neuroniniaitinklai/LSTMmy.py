import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import download
import spacy
from sklearn.utils.class_weight import compute_class_weight

# Atsisiųskite reikalingus NLTK ir spaCy duomenis
download('punkt')
download('stopwords')
nlp = spacy.load("en_core_web_sm")

# Funkcija tekstų valymui
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Pašalinkite nereikalingus simbolius
    doc = nlp(text)
    text = ' '.join([token.lemma_ for token in doc])  # Lematizuokite tekstą
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Pašalinkite stop-words
    return text

# Duomenų paruošimas
data = fetch_20newsgroups(subset="all", remove=('headers', 'footers', 'quotes'))
x_raw = data.data
y = data.target

# Valykite tekstus
x = [clean_text(text) for text in x_raw]

# Klasės kodavimas
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)

# Duomenų atskyrimas
def preprocess_data(x, y, max_vocab_size, max_sequence_length):
    x_train, x_test, y_train, y_test = train_test_split(x, encoded_labels, test_size=0.2, random_state=42)

    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(x_train)

    train_seq = tokenizer.texts_to_sequences(x_train)
    test_seq = tokenizer.texts_to_sequences(x_test)

    x_train_pad = pad_sequences(train_seq, maxlen=max_sequence_length, padding='post', truncating='post')
    x_test_pad = pad_sequences(test_seq, maxlen=max_sequence_length, padding='post', truncating='post')

    num_classes = len(np.unique(encoded_labels))
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    return x_train_pad, x_test_pad, y_train_cat, y_test_cat, num_classes, tokenizer

# Eksperimentų parametrai
sequence_lengths = [50, 100, 200]
vocab_sizes = [20000]

results = {}

# Eksperimentų vykdymas
for max_sequence_length in sequence_lengths:
    for max_vocab_size in vocab_sizes:
        print(f"Mokymas su max_sequence_length={max_sequence_length}, max_vocab_size={max_vocab_size}")

        x_train, x_test, y_train, y_test, num_classes, tokenizer = preprocess_data(x, y, max_vocab_size, max_sequence_length)

        # Modelio sukūrimas
        model = Sequential([
            Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_sequence_length),
            LSTM(64, return_sequences=False),
            Dense(32, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        # Apskaičiuokite klasių svorius
        class_weights = compute_class_weight('balanced', classes=np.unique(encoded_labels), y=encoded_labels)
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

        # Modelio apmokymas
        history = model.fit(
            x_train, y_train,
            validation_split=0.2,
            batch_size=32,
            epochs=100,
            class_weight=class_weight_dict,
            callbacks=[early_stopping],
            verbose=1
        )

        # Testavimo rezultatų vertinimas
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
        print(f"Test accuracy: {test_accuracy:.4f}")

        # Rezultatų saugojimas
        results[(max_sequence_length, max_vocab_size)] = test_accuracy

# Surasti tris geriausius rezultatus
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
print("\nTop 3 eksperimentų parametrai ir jų tikslumas:")
for (params, accuracy) in sorted_results:
    print(f"Parametrai: max_sequence_length={params[0]}, max_vocab_size={params[1]} | Tikslumas: {accuracy:.4f}")

# Rezultatų vizualizacija
fig, ax = plt.subplots(figsize=(12, 8))

for sequence_length in sequence_lengths:
    accuracies = [
        results[(sequence_length, vocab_size)]
        for vocab_size in vocab_sizes
    ]
    ax.plot(
        range(len(accuracies)),
        accuracies,
        marker='o',
        label=f'max_sequence_length={sequence_length}'
    )

ax.set_title("Modelio tikslumas keičiant sekų ilgį ir žodyno dydį")
ax.set_xlabel("Eksperimentai")
ax.set_ylabel("Tikslumas")
ax.legend()
plt.grid()
plt.show()