from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
# Define a tool that performs NER using a pre-trained pipeline
def perform_ner(text: str) -> str:
    from transformers import pipeline
    ner_pipeline = pipeline("ner", grouped_entities=True)
    result = ner_pipeline(text)
    return str(result)
# Wrap the function in a Tool for the agent
ner_tool = Tool(
    name="Named Entity Recognition",
    func=perform_ner,
    description="Extracts entities from the provided text."
)
# Initialize an LLM agent that can use the tool
llm = OpenAI(temperature=0)
agent = initialize_agent(tools=[ner_tool], llm=llm, agent="zero-shot-react-description", verbose=True)
# Use the agent to process a text
text_input = "Elon Musk founded SpaceX in 2002, and Tesla in 2003."
response = agent.run(text_input)
print(response)
