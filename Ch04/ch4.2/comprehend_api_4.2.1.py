import boto3

# Initialize the Comprehend client (make sure AWS credentials are configured)
comprehend = boto3.client('comprehend', region_name='us-east-1')

text = "Barack Obama was born in Hawaii and served as President of the United States."
# Call Amazon Comprehend to detect entities
response = comprehend.detect_entities(Text=text, LanguageCode='en')

# Print out the detected entities
for entity in response['Entities']:
    etext = entity['Text']; etype = entity['Type']; score = entity['Score']
    print(f"{etype}: {etext} (Confidence: {score:.2f})")
