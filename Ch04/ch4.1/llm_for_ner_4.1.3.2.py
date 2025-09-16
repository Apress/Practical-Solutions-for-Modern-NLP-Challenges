import openai
openai.api_key = "YOUR_API_KEY"
text = "Barack Obama visited Paris in July 2017."
prompt = (
    "Identify all named entities in the following text and categorize them:\n"
    f"Text: \"{text}\"\n\n"
    "Format: <Entity> - <Type>\n"
)
response = openai.Completion.create(
    engine="text-davinci-003",  # or a chat model like gpt-3.5-turbo
    prompt=prompt,
    max_tokens=100,
    temperature=0
)
print(response["choices"][0]["text"].strip())
