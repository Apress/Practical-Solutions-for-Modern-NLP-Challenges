# Create a SageMaker model from the trained estimator
predictor = huggingface_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)

# Using the predictor to generate text
prompt_text = "In a world where artificial intelligence has taken over,"
response = predictor.predict({"inputs": prompt_text})

print("Generated Text:", response)
