import string
import scipy
import nltk
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ordered_set import OrderedSet
import pandas as pd
import numpy as np
import re

def get_and_clean_data():
    data = pd.read_json('resources/embold_test.json')
    description = data['title']
    cleaned_description = description.apply(lambda s: s.translate(str.maketrans('', '', string.punctuation + u'\xa0')))
    cleaned_description = cleaned_description.apply(lambda s: s.lower())
    cleaned_description = cleaned_description.apply(lambda s: s.translate(str.maketrans(string.whitespace, ' '*len(string.whitespace), '')))
    cleaned_description = cleaned_description.drop_duplicates()
    return cleaned_description

def create_stem_cache(cleaned_description):
    tokenized_description = cleaned_description.apply(lambda s: word_tokenize(s))
    concated = np.unique(np.concatenate([s for s in tokenized_description.values]))
    stem_cache = {}
    ps = PorterStemmer()
    for s in concated:
        stem_cache[s] = ps.stem(s)
    return stem_cache

def create_custom_preprocessor(stop_dict, stem_cache):
    def custom_preprocessor(s):
        ps = PorterStemmer()
        s = re.sub(r'[^A-Za-z]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = word_tokenize(s)
        s = list(OrderedSet(s)- stop_dict)
        s = [word for word in s if len(word)>2]
        s = [stem_cache[w] if w in stem_cache else ps.stem(w) for w in s]
        s = ' '.join(s)
        return s
    return custom_preprocessor

def sk_vectorize(texts, cleaned_description, stop_dict, stem_cache):
    my_custom_preprocessor = create_custom_preprocessor(stop_dict, stem_cache)
    vectorizer = CountVectorizer(preprocessor=my_custom_preprocessor)
    vectorizer.fit(cleaned_description)
    query = vectorizer.transform(texts)

nltk.download('punkt')
nltk.download('stopwords')
cleaned_description = get_and_clean_data()
stem_cache = create_stem_cache(cleaned_description)
stop_dict = set(stopwords.words('English'))
my_custom_preprocessor = create_custom_preprocessor(stop_dict, stem_cache)
vectorizer_unigram = CountVectorizer(preprocessor=my_custom_preprocessor)
vectorizer_unigram.fit(cleaned_description)
X_unigram = vectorizer_unigram.transform(cleaned_description)
N_unigram = len(cleaned_description)
df_unigram = np.array((X_unigram.todense() > 0).sum(0))[0]
idf_unigram = np.log10(1 + (N_unigram / df_unigram))
tf_unigram = np.log10(X_unigram.todense() + 1)
tf_idf_unigram = np.multiply(tf_unigram, idf_unigram)
X_unigram = scipy.sparse.csr_matrix(tf_idf_unigram)
X_df_unigram = pd.DataFrame(X_unigram.toarray(), columns=vectorizer_unigram.get_feature_names_out())
max_term_unigram = X_df_unigram.sum().sort_values(ascending=False)[:20].index

class BM25(object):
    def __init__(self, vectorizer, b=0.75, k1=1.6):
        self.vectorizer = vectorizer
        self.b = b
        self.k1 = k1

    def fit(self, X):
        """ Fit IDF to documents X """
        self.vectorizer.fit(X)
        self.y = super(TfidfVectorizer, self.vectorizer).transform(X)
        self.avdl = self.y.sum(1).mean()

    def transform(self, q):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        len_y = self.y.sum(1).A1
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        assert sparse.isspmatrix_csr(q)

        y = self.y.tocsc()[:, q.indices]
        denom = y + (k1 * (1 - b + b * len_y / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = y.multiply(np.broadcast_to(idf, y.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1


cleaned_description = get_and_clean_data()
stem_cache = create_stem_cache(cleaned_description)
stop_dict = set(stopwords.words('English'))
my_custom_preprocessor = create_custom_preprocessor(stop_dict, stem_cache)
tf_idf_vectorizer = TfidfVectorizer(preprocessor=my_custom_preprocessor, use_idf=True)
tf_idf_vectorizer.fit(cleaned_description)
bm25 = BM25(tf_idf_vectorizer)
bm25.fit(cleaned_description)


score = bm25.transform('move method')
rank = np.argsort(score)[::-1]
print(cleaned_description.iloc[rank[:5]].to_markdown())

score = bm25.transform('extract method')
rank = np.argsort(score)[::-1]
print(cleaned_description.iloc[rank[:10]].to_markdown())