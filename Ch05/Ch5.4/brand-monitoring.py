# !pip install transformers datasets torch scikit-learn

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Simulated brand-related social media posts
data = {
    "text": [
        "I love the new shoes from BrandX! Super comfy and stylish.",
        "Terrible customer service at BrandX. Never shopping there again.",
        "BrandX has really improved over the last year. Impressed!",
        "The quality of BrandX products is going downhill fast.",
        "BrandX just launched an amazing new product line!",
        "I'm not a fan of BrandX's latest marketing campaign.",
        "Happy with my BrandX purchase. Good value for money.",
        "Disappointed with BrandX delivery delays.",
        "BrandX support team was very helpful and resolved my issue.",
        "Worst experience ever with BrandX. Avoid at all costs!"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# Step 2: Tokenization
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

class BrandSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        encodings = tokenizer(texts, truncation=True, padding=True)
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

# Step 3: Prepare datasets
train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2)
train_dataset = BrandSentimentDataset(train_texts.tolist(), train_labels.tolist(), tokenizer)
val_dataset = BrandSentimentDataset(val_texts.tolist(), val_labels.tolist(), tokenizer)

# Step 4: Load DistilBERT
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

# Step 6: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Step 7: Evaluate model
preds = trainer.predict(val_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
print(classification_report(val_labels, y_pred))
