import pandas as pd
import nltk
nltk.download('wordnet')
nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

data = pd.read_csv('Musical_instruments_reviews.csv')
summary_d = data['summary']

word_tokens = [nltk.word_tokenize(word) for word in summary_d]
porter_stems = [PorterStemmer().stem(word) for word in summary_d]
wordnet_lems = [WordNetLemmatizer().lemmatize(word, 'v') for word in summary_d]

result = zip(word_tokens, porter_stems, wordnet_lems)

for word in result:
    print(f'Tokenize: {word[0]}, Stem: {word[1]}, Lemma: {word[2]}')