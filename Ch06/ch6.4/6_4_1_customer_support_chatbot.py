# Import necessary libraries
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_functions_agent
import openai
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForQuestionAnswering

# Set up OpenAI API key for GPT integration (For LLM-based querying)
openai.api_key = 'your-openai-api-key'

# 1. Load customer support knowledge base
# This could be legal policies, HR documents, technical FAQs, etc.
loader = TextLoader("customer_support_docs.txt")
documents = loader.load()

# 2. Split documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Convert chunks into embeddings using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings()

# Create Chroma vector store for efficient retrieval
vector_store = Chroma.from_documents(chunks, embeddings)

# 4. Implement Extractive QA system with DistilBERT
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased')

def extractive_qa(question):
    inputs = tokenizer(question, chunks[0].page_content, add_special_tokens=True, return_tensors="pt")
    answer_start_scores, answer_end_scores = model(**inputs)
    answer_start = answer_start_scores.argmax()
    answer_end = answer_end_scores.argmax()
    answer = chunks[0].page_content[answer_start: answer_end + 1]
    return answer

# 5. Implement Generative QA with OpenAI GPT (LLMs)
def generative_qa(question):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # or any LLM engine such as GPT-4
        prompt=question,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

# 6. Create a retriever from the vector store
retriever = vector_store.as_retriever()

# 7. Build a RetrievalQA chain using generative and extractive models
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.0),
    chain_type="stuff",
    retriever=retriever
)

# Example: Ask questions and receive answers
question = "How do I reset my password?"
answer = qa_chain.run(question)
print(f"Answer: {answer}")

# 8. Create a Customer Support Chatbot interface
# A simple function to simulate user interaction
def chatbot_interaction(question, use_generative=False):
    if use_generative:
        answer = generative_qa(question)
    else:
        answer = extractive_qa(question)
    return answer

# Simulate Chatbot conversation
user_question = "How do I update my billing information?"
chatbot_answer = chatbot_interaction(user_question, use_generative=True)
print(f"Chatbot Answer: {chatbot_answer}")
