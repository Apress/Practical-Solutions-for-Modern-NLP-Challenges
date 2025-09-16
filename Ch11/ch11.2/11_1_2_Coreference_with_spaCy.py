import spacy
import coreferee

# Load spaCy model and add the coreferee pipeline component
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("coreferee")

# Example text with coreference
doc = nlp("Alice gave Bob a book. He thanked her for the gift.")

# Iterate over coreference chains and print them
for chain in doc._.coref_chains:
    print(chain.pretty_representation)
