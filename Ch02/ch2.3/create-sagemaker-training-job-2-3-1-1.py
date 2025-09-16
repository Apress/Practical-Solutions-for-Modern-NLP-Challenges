from sagemaker.huggingface import HuggingFace

# Hugging Face estimator configuration
huggingface_estimator = HuggingFace(
    entry_point='train.py',            # Your training script
    source_dir='src',                  # Directory with your training code
    base_job_name='huggingface-llm',
    instance_type='ml.p3.2xlarge',     # Example: GPU instance for large models
    instance_count=1,
    transformers_version='4.6',        # Example version
    pytorch_version='1.7',
    py_version='py36',
    role=role,                         # IAM role for SageMaker
    hyperparameters={
        'model_name_or_path': 'gpt2',  # Pre-trained model to fine-tune
        'epochs': 3,
        'train_batch_size': 4,
        'eval_batch_size': 4
    }
)

# Launch the training job
huggingface_estimator.fit({'train': 's3://your-bucket/train-data'})
