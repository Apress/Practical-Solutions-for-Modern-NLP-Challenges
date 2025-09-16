from transformers import pipeline

# Load a small pre-trained generative model (distilled GPT-2 for demonstration)
chat_generator = pipeline("text-generation", model="distilgpt2")
prompt = "User: How can I renew my passport?\nBot:"
result = chat_generator(prompt, max_length=50, num_return_sequences=1)
bot_reply = result[0]['generated_text'].split("Bot:")[-1]  # extract the bot's continuation
print(bot_reply)
