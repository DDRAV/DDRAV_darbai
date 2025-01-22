from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# 1. Duomenų įkėlimas
newsgroups_data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'])
X, y = newsgroups_data.data, newsgroups_data.target

# 2. Duomenų skirstymas į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Tokenizavimas ir embedding išgavimas su BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def extract_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=16)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embedding.flatten())
    return np.array(embeddings)


# 4. Embedding vektorių išgavimas
X_train_embeddings = extract_embeddings(X_train)
X_test_embeddings = extract_embeddings(X_test)

# 5. Logistinės regresijos modelio treniravimas
clf = LogisticRegression(max_iter=100)
clf.fit(X_train_embeddings, y_train)

# 6. Modelio vertinimas
y_pred = clf.predict(X_test_embeddings)
accuracy = accuracy_score(y_test, y_pred)
print(f"Testavimo tikslumas: {accuracy:.2f}")