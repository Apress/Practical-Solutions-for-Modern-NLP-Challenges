from transformers import T5Tokenizer, T5ForConditionalGeneration

context = """Product: Gizmo Pro 3000. Details: The Gizmo Pro 3000 needs to be charged 
for 2 hours before first use. If the device does not turn on, check the battery and 
hold the power button for 5 seconds to reset. The device comes with an 18-month 
warranty from the date of purchase."""

question = "My Gizmo Pro 3000 won't turn on. What should I do?"

# Load a pre-trained T5 model (you could use 't5-base' or a QA fine-tuned version)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Prepare the input by concatenating question and context into a prompt
input_text = f"question: {question}  context: {context}"
inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)

# Generate an answer
generated_ids = model.generate(inputs['input_ids'], max_length=64)
generated_answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_answer)
