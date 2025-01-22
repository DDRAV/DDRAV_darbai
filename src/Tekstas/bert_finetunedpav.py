from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch.nn as nn


# 1. Duomenų įkėlimas
newsgroups_data = fetch_20newsgroups(subset='all', categories=['sci.space', 'comp.graphics'])
X, y = newsgroups_data.data, newsgroups_data.target

# 2. Duomenų skirstymas į mokymo ir testavimo rinkinius
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Sukurkite Dataset objektą Hugging Face formatui
train_dataset = Dataset.from_dict({'text': X_train, 'label': y_train})
test_dataset = Dataset.from_dict({'text': X_test, 'label': y_test})
dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

# 4. Tokenizavimas
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


# 5. Tinkintas modelis su papildomais sluoksniais
class CustomDistilBertModel(nn.Module):
    def __init__(self):
        super(CustomDistilBertModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])


model = CustomDistilBertModel()

# 6. Treniruotės nustatymai
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 7. Modelio apmokymas
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)

trainer.train()

# 8. Modelio vertinimas
results = trainer.evaluate()
print(f"Testavimo tikslumas: {results['eval_accuracy']:.2f}")