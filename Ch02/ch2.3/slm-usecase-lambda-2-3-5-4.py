import os
import json
import boto3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global references for model and tokenizer
model = None
tokenizer = None

S3_BUCKET = "your-model-bucket"
MODEL_KEY = "support-classifier/model.tar.gz"
MODEL_NAME_OR_PATH = "/opt/ml/model"  # Path inside the Lambda layer or container

def download_model_from_s3():
    # Download model artifact if needed (optional approach)
    s3 = boto3.client('s3')
    local_path = '/tmp/model.tar.gz'
    s3.download_file(S3_BUCKET, MODEL_KEY, local_path)
    # Extract or load your model here as needed

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        # If using a package or layer that already has your model files:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_OR_PATH)
        model.eval()

def lambda_handler(event, context):
    # 1. Parse the incoming JSON
    body = json.loads(event["body"])
    ticket_text = body.get("ticket_text", "")

    # 2. Load the model if not already loaded
    load_model()

    # 3. Tokenize and run inference
    inputs = tokenizer(ticket_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # 4. Convert logits to predicted label
    predicted_category_idx = torch.argmax(outputs.logits, dim=1).item()
    
    # Example categories mapping
    categories = ["Billing", "Technical", "Account Management"]
    predicted_category = categories[predicted_category_idx]

    # 5. Build the response
    response_body = {
        "predicted_category": predicted_category
    }

    return {
        "statusCode": 200,
        "body": json.dumps(response_body)
    }
