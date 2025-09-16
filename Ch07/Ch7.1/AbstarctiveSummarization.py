from transformers import pipeline

# Initialize a summarization pipeline with a BART model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
OpenAI's GPT-4 model has shown remarkable performance on a variety of tasks. 
In a recent evaluation, GPT-4 achieved high scores on academic benchmarks and demonstrated the ability to reason about complex problems. 
However, researchers note that the model still lacks transparency in its decision-making process. 
Future work is focusing on improving interpretability and ensuring the model's outputs are trustworthy and free of bias.
"""

summary = summarizer(article, max_length=80, min_length=30, do_sample=False)
print(summary[0]['summary_text'])
