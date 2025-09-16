# Install required packages (transformers, torch, etc.) if not already installed
!pip install transformers torch sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load a T5 model that has been fine-tuned for grammatical error correction
model_name = "abhinavsarkar/Google-T5-base-Grammatical_Error_Correction-Finetuned-C4-200M-550k"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def correct_grammar(input_text):
    # Tokenize the input and run it through the model to generate corrections
    inputs = tokenizer([input_text], return_tensors="pt")
    outputs = model.generate(**inputs, max_length=64, num_beams=4, num_return_sequences=1)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# Example usage:
text = "He are moving here."
print("Original:", text)
print("Corrected:", correct_grammar(text))
