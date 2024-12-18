from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500, stop_words='english')

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    SimpleRNN(64, input_shape=(X_train_rnn.shape[1], 1), activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_rnn, y_train_encoded, epochs=5, batch_size=16, validation_data=(X_test_rnn, y_test_encoded))

test_loss, test_accuracy = model.evaluate(X_test_rnn, y_test_encoded)
print('Loss: ', test_loss, 'Accuracy: ', test_accuracy)