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


def get_user_data(file):
    data = open(file, "r")
    users_data = []
    for line in data:
        (user, country, sex) = line.strip().split(":::")
        users_data.append((user, country, sex))
    data.close()

    return users_data


def get_lexicon_dict(file):
    f = open(file,"r",encoding="utf-8")
    lexicon = {}
    for line in f:
        line = line.strip()
        if line.startswith("#") or len(line) == 0:
            continue
        word,polarity = line.split()
        lexicon[word] = polarity
    f.close()
    return lexicon


def myTokenizer(text):
    tknzr1 = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr1.tokenize(text)


start_time = time.time()


def get_all_tweets(data):
    data_size = len(data)
    total_tweets = 0
    index = 1
    _tweets_text = []
    _user_last_tweet_metadata = []
    for (user, country, sex) in data:
        tweets_file_path = os.path.join(country, user + ".json")
        tweets = open(tweets_file_path, "r")
        i = 0
        user_tweets_text = ""
        tweet_json = {}
        for tweet in tweets:
            i += 1
            tweet_json = json.loads(tweet)
            user_tweets_text += tweet_json["text"] + " "
        # print("user",user,"text",user_tweets_text)
        _tweets_text.append(user_tweets_text)
        _user_last_tweet_metadata.append(tweet_json)
        index_str = str(index) + "/" + str(data_size) + "."
        print(index_str, "user", user, "has", i, "tweets")
        index += 1
        total_tweets += i
        # if index == 201:
        #     break

    print("Total tweets", total_tweets)
    print("Total tweets text", len(_tweets_text))
    return _tweets_text, _user_last_tweet_metadata


def get_extra_features_from_text(size, _tweets_text, _lexicon):
    # add extra features from tweets text:
    # - positive/negative,
    # - number of tokens,
    # - mean size of tokens
    _XX = numpy.zeros((size, 4))
    i = 0
    for text in _tweets_text:
        # print("-")
        tokens = myTokenizer(text)
        positive = 0
        negative = 0
        for token in tokens:
            # print(".",end="")
            sentiment = _lexicon.get(token)
            if sentiment == "positive":
                positive += 1
            elif sentiment == "negative":
                negative += 1
        _XX[i][0] = positive/len(tokens) if positive > 0 else 0
        _XX[i][1] = negative/len(tokens) if negative > 0 else 0
        _XX[i][2] = len(tokens)
        _XX[i][3] = numpy.array([len(x) for x in tokens]).mean() if len(tokens) > 0 else 0
        # print(_XX[i][0]," ",_XX[i][1]," ",_XX[i][2]," ",_XX[i][3])
        i += 1
    print("text extra features size", len(_XX))
    print("text extra features sample", _XX[0:5])
    return _XX


def get_extra_features_from_metadata(size, _user_last_tweet_metadata):
    # add extra features from tweets metadata:
    # - followers count
    # - friends count
    # - favourites count
    # - utc offset
    # - statuses count
    _XX = numpy.zeros((size, 5))
    i = 0
    for metadata in _user_last_tweet_metadata:
        u = metadata.get("user", {})
        followers = u.get("followers_count", 0) if u.get("followers_count", 0) is not None else 0
        friends = u.get("friends_count", 0) if u.get("friends_count", 0) is not None else 0
        favourites = u.get("favourites_count", 0) if u.get("favourites_count", 0) is not None else 0
        utc = u.get("utc_offset", 0) if u.get("utc_offset", 0) is not None else 0
        statuses = u.get("statuses_count", 0) if u.get("statuses_count", 0) is not None else 0
        _XX[i][0] = followers
        _XX[i][1] = friends
        _XX[i][2] = favourites
        _XX[i][3] = utc
        _XX[i][4] = statuses
        # print(_XX[i][0]," ",_XX[i][1]," ",_XX[i][2]," ",_XX[i][3]," ",_XX[i][4])
        i += 1
    print("metadata extra features size", len(_XX))
    print("metadata extra features sample", _XX[0:5])
    return _XX


training_data = get_user_data("training.txt")
training_size = len(training_data)
print("Training size is", training_size)
tweets_text, user_last_tweet_metadata = get_all_tweets(training_data)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))


test_data = get_user_data("test.txt")
test_size = len(test_data)
print("Test size is", test_size)
test_tweets_text, test_user_last_tweet_metadata = get_all_tweets(test_data)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))

norm = None
use_idf = False
min_df = 1
max_df = 1
sublinear_tf = False
smooth_idf = True
max = 1000

vec = CountVectorizer(tokenizer=myTokenizer, max_features=max, ngram_range=(1, 2))
# vec = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
#                       sublinear_tf=sublinear_tf, #stop_words=stopwords.words("spanish"),
#                       max_features=max, tokenizer=myTokenizer, ngram_range=(1, 2))
print("Calculating training features matrix...")
X = vec.fit_transform(tweets_text)
voca = vec.get_feature_names()
print("Features matrix shape is", X.shape)
print("Vocabulary sample", voca)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))


# add extra features
lexicon = get_lexicon_dict("ElhPolar_esV1.lex.txt")


print("Calculating extra features...")
XX = get_extra_features_from_text(X.shape[0], tweets_text, lexicon)
# concatenate features and extra features matrixes
X = numpy.concatenate((X.toarray(), XX), axis=1)

XX = get_extra_features_from_metadata(X.shape[0], user_last_tweet_metadata)
# concatenate features and extra features matrixes
X = numpy.concatenate((X, XX), axis=1)

print("The final features matrix has shape", X.shape)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))



print("Calculating test features matrix...")
test_X = vec.transform(test_tweets_text)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))

print("Calculating test extra features...")
test_XX = get_extra_features_from_text(test_X.shape[0], test_tweets_text, lexicon)
# concatenate features and extra features matrixes
test_X = numpy.concatenate((test_X.toarray(), test_XX), axis=1)

test_XX = get_extra_features_from_metadata(test_X.shape[0], test_user_last_tweet_metadata)
# concatenate features and extra features matrixes
test_X = numpy.concatenate((test_X, test_XX), axis=1)

print("The final test features matrix has shape", test_X.shape)
print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))

y = numpy.array([country for (user, country, sex) in training_data][0:])
print("Target",y)

# clf = svm.SVC()
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

preds = clf.predict(test_X)
print("Predictions",preds)

f = open("preds"+str(time.time())+".txt","w",encoding="utf-8")
for p in preds:
    f.write(p + ";")
f.write("\n")
f.close()

test_y = numpy.array([country for (user, country, sex) in test_data][0:])

print("Test target",test_y)
print("There are ", (test_y != preds).sum(), "wrong predictions out of", len(test_y))
print("Accuracy:", (100.0 * (test_y == preds).sum()) / test_X.shape[0])

print("--- %s seconds ---" % (time.time() - start_time))
print("--- %.2f minutes ---" % ((time.time() - start_time)/60))


f = open("results"+str(time.time())+".txt","w",encoding="utf-8")
for v in voca:
    f.write(v + ";")
f.write("\n")
for x in X:
    for xx in x:
        f.write(str(xx) + ";")
    f.write("\n")
f.close()


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