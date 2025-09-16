from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. Load a pre-trained model and tokenizer (DistilBERT base uncased in this example)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: positive and negative

# 2. Prepare a small custom training dataset (for example purposes)
train_texts = [
    "Absolutely fantastic! The new update completely exceeded my expectations.",  # should be positive
    "Terrible experience. The product stopped working after a week.",             # should be negative
    "I am not sure if I like it or not.",                                        # maybe neutral, but we'll label for binary
    "Really good value for the money.",                                          # positive
]
train_labels = [1, 0, 0, 1]  # Let's use 1 for positive, 0 for negative for binary classification

# Tokenize the training texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

# Create a Dataset object
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = ReviewsDataset(train_encodings, train_labels)

# 3. Set up training arguments
training_args = TrainingArguments(
    output_dir="sentiment_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_steps=10,
    logging_dir="logs",
    no_cuda=True  # set to False if you have a GPU
)

# 4. Initialize Trainer with our model, data, and training configurations
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 5. Fine-tune the model
trainer.train()
