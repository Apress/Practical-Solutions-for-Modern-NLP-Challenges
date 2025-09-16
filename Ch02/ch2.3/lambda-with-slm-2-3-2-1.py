import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global model references (loaded once per container)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

def lambda_handler(event, context):
    # Ensure the model is loaded
    load_model()

    # Parse incoming data
    body = json.loads(event["body"])
    text_input = body.get("text", "")

    # Tokenize and predict
    inputs = tokenizer(text_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Convert logits to predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

    # Construct response
    response = {
        "statusCode": 200,
        "body": json.dumps({"predicted_class": int(predicted_class)})
    }
    return response
