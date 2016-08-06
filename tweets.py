import os
import json
import re
import numpy
import sys

from prettytable import PrettyTable

from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.classify import NaiveBayesClassifier

from collections import Counter
from operator import add

from decimal import Decimal

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier

import time

def getUserData(file):
    data = open(file, "r")
    # lines = data.read().splitlines()

    dict = {}
    for line in data:
        (user, country, sex) = line.strip().split(":::")
        dict.update([(user, (country, sex))])

    data.close()

    return dict


def getLexiconDict(file):
    f = open(file,"r",encoding="utf-8")
    lexicon = {}
    for line in f:
        line = line.strip()
        if line.startswith("#") or len(line) == 0:
            continue
        # print(line.split())
        word,polarity = line.split()
        lexicon[word] = polarity
    f.close()
    return lexicon


def myTokenizer(text):
    tknzr1 = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr1.tokenize(text)


start_time = time.time()


def getAllTweets(data):
    data_size = len(data)
    total_tweets = 0
    index = 1
    tweets_text = []
    for (user, (country, sex)) in data.items():
        tweets_file_path = os.path.join(country, user + ".json")
        tweets = open(tweets_file_path, "r")
        i = 0
        user_tweets_text = ""
        for tweet in tweets:
            i += 1
            tweet_json = json.loads(tweet)
            user_tweets_text += tweet_json["text"] + " "
        tweets_text.append(user_tweets_text)
        index_str = str(index) + "/" + str(data_size) + "."
        print(index_str, "user", user, "has", i, "tweets")
        index += 1
        total_tweets += i
        # if index == 101:
        #     break

    print("Total tweets", total_tweets)
    print("Total tweets text", len(tweets_text))
    return tweets_text


training_data = getUserData("training.txt")
training_size = len(training_data)
print("Training size is", training_size)
tweets_text = getAllTweets(training_data)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))



test_data = getUserData("test.txt")
test_size = len(test_data)
print("Test size is", test_size)
test_tweets_text = getAllTweets(test_data)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))

norm = None
use_idf = True
min_df = 1
max_df = 1
sublinear_tf = False
max = 500

vec = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf,
                      stop_words=stopwords.words("spanish"),
                      max_features=max, tokenizer=myTokenizer, ngram_range=(1, 1))
print("Calculating training features matrix...")
X = vec.fit_transform(tweets_text)
voca = vec.get_feature_names()
print("Features matrix shape is", X.shape)
print("Vocabulary sample", voca[0:50])
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))


# add extra features
lexicon = getLexiconDict("ElhPolar_esV1.lex.txt")

def getExtraFeaturesSentiments(size, tweets_text, lexicon):
    XX = numpy.zeros((size, 2))
    # add extra features
    i = 0
    for text in tweets_text:
        tokens = myTokenizer(text)
        positive = 0
        negative = 0
        for token in tokens:
            sentiment = lexicon.get(token)
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1
        XX[i][0] = positive/len(tokens) if positive > 0 else 0
        XX[i][1] = negative/len(tokens) if negative > 0 else 0
        i += 1
    return XX


print("Calculating extra features...")
XX = getExtraFeaturesSentiments(X.shape[0], tweets_text, lexicon)
print("extra features size", len(XX))
print("extra features sample", XX[0:5])
# concatenate features and extra features matrixes
X = numpy.concatenate((X.toarray(), XX), axis=1)
print("The final features matrix has shape", X.shape)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))



print("Calculating test features matrix...")
test_X = vec.transform(test_tweets_text)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))

print("Calculating test extra features...")
test_XX = getExtraFeaturesSentiments(test_X.shape[0], test_tweets_text, lexicon)
print("test extra features has shape", test_XX.shape)
print("test extra features sample", test_XX[0:5])
# concatenate features and extra features matrixes
test_X = numpy.concatenate((test_X.toarray(), test_XX), axis=1)
print("The final test features matrix has shape", test_X.shape)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))


y = numpy.array([sex for (user, (country, sex)) in training_data.items()][0:])

clf = svm.SVC()
clf.fit(X, y)

preds = clf.predict(test_X)
print(preds)

test_y = numpy.array([sex for (user, (country, sex)) in test_data.items()][0:])

print(test_y)
print("There are ", (test_y != preds).sum(), "wrong predictions out of", len(test_y))
print("Accuracy:", (100.0 * (test_y == preds).sum()) / test_X.shape[0])

print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))


# TODO:
# check most used words by class
# clean dictionaries (remove http, https, ...)
# improve dictionaries (existance of domains, eg: .co)
# combination between model and feature engineering
# graphical analysis:
## mean followers by country
## mean hashtags by user
## mean word size by user
## mean tweet size by user
# use sentiment