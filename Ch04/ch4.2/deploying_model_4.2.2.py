predictor = huggingface_estimator.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')
# The predictor is now a live endpoint that we can invoke with text
result = predictor.predict({"inputs": "Apple Inc. is headquartered in Cupertino, California."})
print(result)
