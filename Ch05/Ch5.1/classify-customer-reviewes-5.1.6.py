from transformers import pipeline

# Load a sentiment analysis pipeline (this downloads a pre-trained DistilBERT model under the hood)
classifier = pipeline("sentiment-analysis")

# Example reviews to analyze
reviews = [
    "I bought this laptop two weeks ago, and I am absolutely delighted with its performance!",
    "The phone case broke after two days... really poor quality.",
    "Not bad at all – the product was actually better than I expected.",
]

# Run the sentiment classifier on each review
results = classifier(reviews)
for review, result in zip(reviews, results):
    print(f"Review: {review}\nPredicted Sentiment: {result['label']} (score={result['score']:.3f})\n")
