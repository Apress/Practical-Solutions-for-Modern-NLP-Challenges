from transformers import pipeline
# Load a pre-trained NER pipeline (this will download a model like BERT fine-tuned on NER)
ner_pipeline = pipeline("ner", grouped_entities=True)
result = ner_pipeline("Google was founded by Larry Page and Sergey Brin in California.")
print(result)
