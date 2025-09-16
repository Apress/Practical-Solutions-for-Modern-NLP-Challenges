# Load the fine-tuned model from the output directory
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("sentiment_model")
fine_tuned_classifier = pipeline("sentiment-analysis", model=fine_tuned_model, tokenizer=tokenizer)

print(fine_tuned_classifier("The overall experience was decent, not great but okay."))
