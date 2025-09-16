# Install the summarizer library (if not already installed)
!pip install bert-extractive-summarizer transformers

from summarizer import Summarizer

text = """
The COVID-19 pandemic caused unprecedented disruption to global supply chains. 
Many industries faced shortages of critical components, leading to production delays. 
In response, companies are re-evaluating their supply chain strategies, with a focus on resilience and localization. 
Experts predict that these changes could permanently reshape international trade.
"""

model = Summarizer()  # This loads a distilled BERT model by default for efficiency
summary = model(text, num_sentences=2)  # Extract 2 most important sentences
print(summary)
