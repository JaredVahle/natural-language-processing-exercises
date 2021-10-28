import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

def basic_clean(article):
    article = article.lower()
    article = unicodedata.normalize('NFKD', article).encode('ascii','ignore').decode('utf-8')
    article = re.sub(r"[^a-z0-9'\s]", '', article)
    return article

def tokenize(article):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    article = tokenizer.tokenize(article, return_str = True)
    return article, tokenizer

def stem(article):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in article.split()]
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lammatize(article):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in article.split()]
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized
    
def remove_stopwords(article,extra_words = [], exclude_words = []):
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    stopword_list = stopword_list.union(set(extra_words))

    words = article.split()
    filtered_words = [word for word in words if word not in stopword_list]
    article_without_stopwords = ' '.join(filtered_words)
    return article_without_stopwords

def prep_article_data(df,content,extra_words=[],exclude_words=[]):
    df['title'] = df.title

    df['original'] = df[content]
    df['clean'] = df[content].apply(basic_clean).apply(tokenize).apply(lambda x: remove_stopwords(x, extra_words, exclude_words))
    df['stemmed'] = df['clean'].apply(stem)
    df['lemmatized'] = df['clean'].apply(lemmatize)
    return df[['title','original','clean','stemmed','lemmatized']]