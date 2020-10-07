import re
import numpy as np
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup as bs


def get_stem_tokens(text):
    text = text.lower()
    text = re.sub(r'[-+]?\d+', '', text)
    text = text.translate((str.maketrans('', '', string.punctuation)))
    text = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if not word in stop_words]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text]
    return text


def get_text_list_from_xml(xmlContent, n=1200, remove_links=False):
    text_list = []
    soup = bs(xmlContent, "lxml")
    for row in soup.find_all("row")[:n]:
        soup_in = bs(row.get("body"), "lxml")
        if remove_links:
            for a in soup_in.findAll('a', href=True):
                a.extract()
        text_list.append(soup_in.get_text())

    return text_list


def embeding_from_stem(vocab_map, list_stems, ignore_unk=True, normalize=False):
    # list_stems = [x for x in list_stems if x]
    embeddings = np.zeros((len(list_stems), len(vocab_map)))

    for i, embed in enumerate(embeddings):
        for s in list_stems[i]:
            if ignore_unk and s not in vocab_map.keys():
                continue
            embed[vocab_map[s]] += 1

    return embeddings


def tf_idf(X, alpha=0, beta=0):
    tf = X / X.sum(axis=1, keepdims=True)
    idf = np.log((X.shape[0] + alpha) / ((X > 0).sum(axis=0) + beta))
    return tf * idf
