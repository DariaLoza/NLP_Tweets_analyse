#!pip install -U spacy
#Download Language Model
#!python -m spacy download en_core_web_sm
import spacy
from spacy import displacy
from collections import Counter
import pandas as pd
pd.options.display.max_rows = 600
pd.options.display.max_colwidth = 400

#Load Language Model
import en_core_web_sm
nlp = en_core_web_sm.load()

#Import data
filepath = "preparing_tweets_boomers.txt"
text = open(filepath, encoding='utf-8').read()
document = nlp(text)

#displacy.render(document, style="ent")
displacy.render(document, style='dep', jupyter=True, options={'distance': 90})


#Get Named Entities

document.ents

#For each named_entity in document.ents, we will extract the named_entity and its corresponding named_entity.label_.
for named_entity in document.ents:
    print(named_entity, named_entity.label_)

#To extract just the named entities that have been identified as PERSON, we can add a simple if statement into the mix
for named_entity in document.ents:
    if named_entity.label_ == "PERSON":
        print(named_entity)



#NER with Long Texts

filepath = "preparing_tweets_boomers.txt"
text = open(filepath).read()

import math
number_of_chunks = 80

chunk_size = math.ceil(len(text) / number_of_chunks)

text_chunks = []

for number in range(0, len(text), chunk_size):
    text_chunk = text[number:number+chunk_size]
    text_chunks.append(text_chunk)

chunked_documents = list(nlp.pipe(text_chunks))

#Get People
people = []

for document in chunked_documents:
    for named_entity in document.ents:
        if named_entity.label_ == "PERSON":
            people.append(named_entity.text)

people_tally = Counter(people)

df = pd.DataFrame(people_tally.most_common(), columns=['character', 'count'])
print(df)

#Get Places

places = []
for document in chunked_documents:
    for named_entity in document.ents:
        if named_entity.label_ == "GPE" or named_entity.label_ == "LOC":
            places.append(named_entity.text)

places_tally = Counter(places)

df = pd.DataFrame(places_tally.most_common(), columns=['place', 'count'])
print(df)



