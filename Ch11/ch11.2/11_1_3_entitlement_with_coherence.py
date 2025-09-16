from transformers import pipeline

# Load a pre-trained entailment model
classifier = pipeline("text-classification", model="roberta-large-mnli")

premise = "The cat is on the mat."
hypothesis = "The animal is on the mat."
result = classifier(f"{premise} {hypothesis}")

print(result)
