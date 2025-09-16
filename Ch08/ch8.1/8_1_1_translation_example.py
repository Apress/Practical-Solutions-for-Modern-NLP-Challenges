from transformers import pipeline

# Load a small translation model (English to French)
translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

text = "Hello, how are you? I am excited to learn new things!"
result = translator(text, max_length=60)[0]['translation_text']
print(result)
