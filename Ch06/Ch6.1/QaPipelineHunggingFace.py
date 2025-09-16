from transformers import pipeline

context = """Product: Gizmo Pro 3000. Details: The Gizmo Pro 3000 needs to be charged 
for 2 hours before first use. If the device does not turn on, check the battery and 
hold the power button for 5 seconds to reset. The device comes with an 18-month 
warranty from the date of purchase."""

question = "How long is the warranty for the Gizmo Pro 3000?"

# Load a QA pipeline with an extractive model (e.g., DistilBERT fine-tuned on SQuAD)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

result = qa_pipeline(question=question, context=context)
print(result['answer'])
