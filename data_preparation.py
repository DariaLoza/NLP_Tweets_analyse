import string
import re

import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize, PorterStemmer

with open('parsing_data_boomers.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.lower()
    spec_chars = string.punctuation + '\n\xa0«»\t-...'
    text = "".join([ch for ch in text if ch not in spec_chars])
    text = re.sub('\n', '', text)


    def remove_chars_from_text(text, chars):
        return "".join([ch for ch in text if ch not in chars])


    text = remove_chars_from_text(text, spec_chars)
    text = remove_chars_from_text(text, string.digits)

tokens = word_tokenize(text)
tokens = [w.lower() for w in tokens]

table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in tokens]

words = [word for word in stripped if word.isalpha()]

stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
porter = PorterStemmer()
data_stemmed = [porter.stem(word) for word in tokens]
