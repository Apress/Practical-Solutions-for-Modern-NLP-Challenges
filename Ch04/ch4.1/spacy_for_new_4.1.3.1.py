import spacy
# Load a small English NER model in spaCy
nlp = spacy.load("en_core_web_sm")
# Sample text
text = "Google was founded by Larry Page and Sergey Brin in California."
# Process the text through the pipeline
doc = nlp(text)
# Extract and print entities
for ent in doc.ents:
    print(ent.text, ent.label_)
