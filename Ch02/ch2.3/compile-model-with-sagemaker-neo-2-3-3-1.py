import boto3
import sagemaker
from sagemaker.pytorch.model import PyTorchModel
from sagemaker.amazon.neo import NeoCompilationJob

# Assume we already have a trained PyTorch model in S3
model_data = "s3://path-to-your-model/model.tar.gz"

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"

# Create a PyTorchModel object from your trained model
pytorch_model = PyTorchModel(
    model_data=model_data,
    role=role,
    framework_version="1.7",
    py_version="py3"
)

# Create a Neo compilation job
neo_compilation_job = NeoCompilationJob(
    role=role,
    input_model=pytorch_model,
    target_instance_family="ml_c5",  # Example target
    output_path="s3://your-output-bucket/compiled-model/",
    framework="PYTORCH",
    framework_version="1.7",
    job_name="text-gen-neo-compile"
)

# Launch the compilation
neo_compilation_job.compile_model()
