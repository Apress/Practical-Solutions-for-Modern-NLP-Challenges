import boto3
import sagemaker
from sagemaker import get_execution_role

# Initialize a session using Boto3
session = boto3.Session(
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY',
    region_name='us-east-1'
)

# Create a SageMaker session object
sagemaker_session = sagemaker.Session(
    boto_session=session
)

# Obtain the execution role for SageMaker (if running inside SageMaker Notebook)
role = get_execution_role()

print("SageMaker session initialized and role obtained.")
# Note: In practice, you would configure AWS credentials and roles using best practices such as IAM roles, AWS CLI profiles, or environment variables rather than hardcoding keys.