from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import datasets

# For illustration: hyperparameters are passed via command-line args or environment in SageMaker
model_name = hyperparams.get("model_name", "bert-base-cased")
num_labels = hyperparams.get("num_labels", 5)

# Load tokenizer and model from pre-trained checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# Load our dataset (assuming we have prepared it in a CSV or JSON and uploaded to the container)
train_dataset = datasets.load_dataset('json', data_files='train.json', split='train')
val_dataset = datasets.load_dataset('json', data_files='val.json', split='train')

# Tokenize and align labels function...
# (Convert words to subword tokens, align entity labels to tokens, etc.)

train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

training_args = TrainingArguments(
    output_dir="/opt/ml/model",  # SageMaker will save model here
    num_train_epochs=hyperparams.get("epochs", 3),
    per_device_train_batch_size=hyperparams.get("train_batch_size", 16),
    learning_rate=hyperparams.get("learning_rate", 5e-5),
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # define compute_metrics to compute F1/accuracy from predictions if needed
)

trainer.train()
trainer.save_model("/opt/ml/model")  # Save the final model to output directory
