import boto3
import json

runtime = boto3.client('runtime.sagemaker', region_name='us-east-1')

# Example prompt for the LLM
prompt_text = (
    "Write a short social media post highlighting the eco-friendly features "
    "of our new electric bicycle. Tone: enthusiastic, brand voice: friendly."
)

# Create the JSON payload for the endpoint
payload = {
    "inputs": prompt_text,
    # You can also pass inference parameters, e.g., max_length, temperature, etc.
    "parameters": {
        "max_length": 100,
        "temperature": 0.7
    }
}

# Invoke the endpoint
response = runtime.invoke_endpoint(
    EndpointName="marketing-llm-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload)
)

# Parse the response
result = json.loads(response["Body"].read())

# Assuming Hugging Face JSON output structure
generated_text = result[0]["generated_text"]
print("Generated Marketing Copy:", generated_text)
