import sagemaker
from sagemaker.huggingface import HuggingFace

# Define hyperparameters for fine-tuning
hyperparams = {
    'model_name': 'bert-base-cased',      # pre-trained model
    'num_labels': 5,                     # e.g., our custom entity types count
    'epochs': 3,
    'train_batch_size': 16,
    'learning_rate': 2e-5
}

huggingface_estimator = HuggingFace(
    entry_point='train.py',
    source_dir='src',  # directory with train.py and perhaps data processing code
    instance_type='ml.p3.2xlarge',
    instance_count=1,
    role=sagemaker_role,
    transformers_version='4.26',    # matching versions for the HF DLC
    pytorch_version='1.13',
    py_version='py39',
    hyperparameters=hyperparams
)

# Launch the training job on SageMaker
huggingface_estimator.fit({'train': train_s3_uri, 'validation': val_s3_uri})
