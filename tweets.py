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

MAX_FEATURES = 1000

SUPERCLASSES = ["Country","Sex"]

GENDERS = ["male","female","UNKNOWN"]

GENDER_WORDS = {"male":["futbol","gol","jugador","tio"],
                "female":["casa","comida","peso","yo","llorar","triste"],
                "UNKNOWN":[""]}


COUNTRIES = ["colombia","mexico","venezuela","espana","peru","chile","argentina"]

COUNTRY_WORDS = {"colombia":["colombia","carro","harto","usted","ustedes"],
                 "mexico":["mexico","chinga","chingar","guey","wey","chilli","chili","acapulco"],
                 "venezuela":["venezuela"],
                 "espana":["espana","tÃ­o","tio","tÃ­a","tia","coche"],
                 "peru":["peru"],
                 "chile":["chile"],
                 "argentina":["argentina","boludo","pelotudo","concha"]}


def pinta_matriz_dispersa(M, nombre_col=None, pre=2):
    filas, columnas = M.shape
    header = nombre_col != None
    pt = PrettyTable(nombre_col, header=header)
    for fila in range(filas):
        vf = M.getrow(fila)
        _, cind = vf.nonzero()
        #f = [vf[0, c] if c in cind else '-' for c in range(columnas)]
        pt.add_row([round(vf[0, c],pre) if c in cind else '-' for c in range(columnas)])
        #print (f)
        #pt.add_row(f)
    return pt

def saveDictionary(path, filename, dict):
    os.makedirs(path, exist_ok=True)
    filePath = os.path.join(path, filename)
    # print("Saving dictionary in file " + filePath)
    file = open(filePath, "w", encoding="utf-8")
    file.write(json.dumps(dict,default=str))
    file.close()


def recoverDictionary(file):
    dictFile = open(file, "r")
    line = dictFile.readline()
    dict = json.loads(line)
    dictFile.close()
    return dict


def saveMostCommonList(path, file, mostCommon):
    os.makedirs(path, exist_ok=True)
    filePath = os.path.join(path, file)
    file = open(filePath, "w", encoding="utf-8")
    for (w, count) in mostCommon:
        file.write(w + "|")
    file.write("\n")
    for (w, count) in mostCommon:
        file.write(str(count) + "|")
    file.write("\n")
    file.close()


def getUserData(file):
    data = open(file, "r")
    # lines = data.read().splitlines()

    dict = {}
    for line in data:
        (user, country, sex) = line.strip().split(":::")
        dict.update([(user, (country, sex))])

    data.close()

    return dict


def mergeDictionaries(a,b):
    A = Counter(a)
    B = Counter(b)
    return A+B


def myTokenizer(text):
    tknzr1 = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr1.tokenize(text)


def myVectorizer(docs, max):
    norm = None
    use_idf = True
    min_df = 1
    max_df = 1
    sublinear_tf = False
    # vec = CountVectorizer(tokenizer=myTokenizer)
    # vec = TfidfVectorizer(norm=None, smooth_idf=False, stop_words=stop)
    # vec = TfidfVectorizer(norm=None, use_idf=False, stop_words=stopwords.words("spanish"), max_features=max, tokenizer=myTokenizer)
    # vec = TfidfVectorizer(norm=None, use_idf=False, max_features=max, tokenizer=myTokenizer)
    # vec = TfidfVectorizer(norm=None, use_idf=False)
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf,
                          stop_words=stopwords.words("spanish"),
                          max_features=max, tokenizer=myTokenizer, ngram_range=(1, 2))
    X = vec.fit_transform(docs)
    voca = vec.get_feature_names()
    return X, voca



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
use_idf = False
min_df = 1
max_df = 1
sublinear_tf = False
max = 2000

vec = TfidfVectorizer(min_df=min_df, max_df=max_df, norm=norm, use_idf=use_idf, sublinear_tf=sublinear_tf,
                      stop_words=stopwords.words("spanish"),
                      max_features=max, tokenizer=myTokenizer, ngram_range=(1, 2))
print("Calculating training features matrix...")
X = vec.fit_transform(tweets_text)
voca = vec.get_feature_names()
print("Features matrix shape is", X.shape)
print("Vocabulary is", voca)

print("Calculating test features matrix...")
test_X = vec.transform(test_tweets_text)
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

exit()

def generateWordDictionary(docs):
    X, voca = myVectorizer(docs, MAX_FEATURES)

    aggregateCols = X.sum(axis=0)
    # aggregateCols = X.mean(axis=0)

    mydict = dict()
    i = 0
    for w in voca:
        wordCount = aggregateCols.getA()[0][i]
        count = mydict.get(w, 0) + wordCount
        mydict.update( [(w, count)] )
        i += 1

    return mydict


def getUserJSONDocs(user, country, max=0):
    userFile = os.path.join(country, user + ".json")
    f = open(userFile, "r")
    tweets = f.read().splitlines()

    if len(tweets) == 0:
        return []

    docs = []
    i = 0
    for t in tweets:
        js = json.loads(t)
        docs.append(js)
        i += 1
        if i == max:
            break

    return docs


def getUserTweets(user,country):
    docs = []
    jsonDocs = getUserJSONDocs(user,country)
    for js in jsonDocs:
        text = js["text"]
        docs.append(text)

    return docs


def getUserDict(user, country):
    dictFilePath = os.path.join("userDicts", user + ".json")
    userDict = {}
    if (os.path.exists(dictFilePath)):
        # already calculated
        userDict = recoverDictionary(dictFilePath)
        for key, value in userDict.items():
            userDict[key] = int(value)
    else:
        # must be calculated
        tweets = getUserTweets(user,country)
        if len(tweets) > 0:
            userDict = generateWordDictionary(tweets)
            saveDictionary("userDicts", user + ".json", userDict)

    return userDict


def getMostCommonWordsByField(field,file):
    print("Getting common words for " + field)
    totalDictFilePath = os.path.join("generated","total_dictionary_" + field + ".txt")
    if(os.path.exists(totalDictFilePath)):
        # already calculated
        print("File " + totalDictFilePath + " exists")
        totalDict = recoverDictionary(totalDictFilePath)
    else:
        print("File " + totalDictFilePath + " doesn't exist. Most common words will be calculated.")
        # must be calculated
        users = getUserData(file)
        docs = []
        i = 1
        for (user, (country, sex)) in users.items():
            if i%50 == 0:
                print("Most common words by field " + field + ": Processing " + str(i) + "/" + str(len(users)))
            i += 1
            if country == field or sex == field:
                tweets = getUserTweets(user,country)
                for tweet in tweets:
                    docs.append(tweet)

        print("Generating word dictionary for " + field + " using " + str(len(docs)) + " docs")
        totalDict = generateWordDictionary(docs)

        saveDictionary("generated","total_dictionary_"+field+".txt",totalDict)

    mostCommon = Counter(totalDict).most_common(MAX_FEATURES)

    list = []
    for (w, count) in mostCommon:
        list.append(w)
    print("First 100 most common words for " + field + ": ")
    print(list[0:100])

    print("Saving most common words list for " + field)
    saveMostCommonList("generated", "most_common_" + field + ".csv", mostCommon)

    return mostCommon


def getMostCommonWords(file):
    print("Getting total common words ")
    totalDictFilePath = os.path.join("generated","total_dictionary_" + file)
    if(os.path.exists(totalDictFilePath)):
        # already calculated
        print("File " + totalDictFilePath + " exists")
        totalDict = recoverDictionary(totalDictFilePath)
    else:
        print("File " + totalDictFilePath + " doesn't exist. Most common words will be calculated.")
        # must be calculated
        users = getUserData(file)
        docs = []
        i = 1
        for (user, (country, sex)) in users.items():
            if i%50 == 0:
                print("Most common words: Processing " + str(i) + "/" + str(len(users)))
            i += 1
            tweets = getUserTweets(user,country)
            for tweet in tweets:
                docs.append(tweet)

        print("Generating total word dictionary for " + file + " using " + str(len(docs)) + " docs")
        totalDict = generateWordDictionary(docs)

        saveDictionary("generated","total_dictionary_"+file,totalDict)

    mostCommon = Counter(totalDict).most_common(MAX_FEATURES)
    list = []
    for (w, count) in mostCommon:
        list.append(w)
    print("First 100 most common words: ")
    print(list[0:100])

    print("Saving most common words list")
    saveMostCommonList("generated", "most_common.csv", mostCommon)

    return mostCommon


def getMostCommonWordsBySex(file):
    print("Getting most common words by sex")
    genderDict = dict()
    for gender in GENDERS:
        mostCommon = getMostCommonWordsByField(gender,file)
        mostCommonDict = dict(mostCommon)
        genderDict = mergeDictionaries(genderDict, mostCommonDict)
    return genderDict.most_common(MAX_FEATURES)


def getMostCommonWordsByCountry(file):
    print("Getting most common words by country")
    countryDict = dict()
    for country in COUNTRIES:
        mostCommon = getMostCommonWordsByField(country,file)
        mostCommonDict = dict(mostCommon)
        countryDict = mergeDictionaries(countryDict, mostCommonDict)
    return countryDict.most_common(MAX_FEATURES)


def createMatrix(file,classIndex):
    print("Creating matrix for " + file + " and class " + SUPERCLASSES[classIndex])
    users = getUserData(file)
    mostCommon = getMostCommonWordsByCountry(file) if classIndex == 0 else getMostCommonWordsBySex(file)
    words = [w for (w,count) in mostCommon]
    features = list.copy(words)
    # features = list(range(len(words)))
    X = []
    y = []
    i = 1
    print("Create Matrix: Processing " + str(len(users)))
    for (user,(country,sex)) in users.items():
        if i%50 == 0:
            print(str(i) + " ", end="")
            sys.stdout.flush()
        i += 1
        userDict = getUserDict(user,country)

        wordCounts = []
        for word in words:
            wordCount = userDict.get(word, 0)
            wordCounts.append(wordCount)

        # mean word size
        meanWordSize = 0
        maxWordSize = 0
        if len(userDict) > 0:
            meanWordSize = sum( [len(word) for (word,count) in userDict.items()] ) / len(userDict)
            maxWordSize = max( [len(word) for (word,count) in userDict.items()] )
        wordCounts.append(meanWordSize)
        wordCounts.append(maxWordSize)

        # mean tweet size
        meanTweetSize = 0
        maxPhraseSize = 0
        tweets = getUserTweets(user,country)
        if len(tweets) > 0:
            meanTweetSize = sum([len(x) for x in tweets]) / len(tweets)
            maxPhraseSize = max([len(x) for x in tweets])
        wordCounts.append(meanTweetSize)
        wordCounts.append(maxPhraseSize)

        # add user info
        jsonDocs = getUserJSONDocs(user,country,1)
        followers = 0
        friends = 0
        favourites = 0
        utc = 0
        statuses = 0

        if len(jsonDocs) > 0:
            u = jsonDocs[0]["user"]
            followers = u["followers_count"] if u["followers_count"] is not None else 0
            friends = u["friends_count"] if u["friends_count"] is not None else 0
            favourites = u["favourites_count"] if u["favourites_count"] is not None else 0
            utc = u["utc_offset"] if u["utc_offset"] is not None else 0
            statuses = u["statuses_count"] if u["statuses_count"] is not None else 0
        wordCounts.append(followers)
        wordCounts.append(friends)
        wordCounts.append(favourites)
        wordCounts.append(utc)
        wordCounts.append(statuses)

        # features
        X.append(wordCounts)
        # classes
        y.append([country,sex])

    # extra features names
    features.append("mean word size")
    features.append("max word size")
    features.append("mean tweet size")
    features.append("max phrase size")
    features.append("followers")
    features.append("friends")
    features.append("favourites")
    features.append("utc")
    features.append("statuses")

    features.append("country")
    features.append("sex")

    X = numpy.array(X)
    y = numpy.array(y)

    print("")
    print(X.shape)
    print(y.shape)
    print(len(features))
    return X,y,features


def trainAndPredict(X, y, test_X, test_y, classIndex):
    # clf = tree.DecisionTreeClassifier()
    clf = RandomForestClassifier(n_estimators=500)
    # clf = svm.SVC()
    # clf = GaussianNB()
    # clf = linear_model.LogisticRegression()

    print(SUPERCLASSES[classIndex] + ": Learning...")
    print(X.shape)
    print(y.shape)
    clf = clf.fit(X, y[:,classIndex])

    print(SUPERCLASSES[classIndex] + ": Predicting...")
    preds = clf.predict(test_X)

    predsFile = open("preds_"+SUPERCLASSES[classIndex]+".txt","w")
    for pred in preds:
        predsFile.write(str(pred) + "\n")
    predsFile.close()

    print(test_y[:,classIndex])
    print(preds)
    print( SUPERCLASSES[classIndex] + ": %d muestras mal clasificadas de %d" % ( (test_y[:,classIndex] != preds).sum(), len(test_y[:,classIndex]) ) )
    print( SUPERCLASSES[classIndex] + ": Accuracy = %.1f%%" % ( ( 100.0 * (test_y[:,classIndex] == preds).sum() ) / len(X) ) )


def saveMatrix(X, y, features, file):
    print("Saving matrix " + file)
    print(X.shape)
    print(y.shape)
    print(len(features))
    filePath = os.path.join("generated","matrix_"+file+".csv")
    file = open(filePath,"w",encoding="utf-8")

    i = 0
    for word in features:
        if "," in word:
            word = word.replace(",", "comma")
        # file.write("'" + str(i) + "-" + str(word) + "',")
        file.write("'" + str(i) + "',")
        i += 1
    file.write("\n")

    print(str(i) + " features written")
    i = 0
    for xs in X:
        j = 0
        for xsi in xs:
            file.write("'" + str(xsi) + "',")
            j += 1
        # print(str(j) + " data values written ", end="")
        j = 0
        for ys in y[i]:
            file.write("'" + str(ys) + "',")
            j += 1
        file.write("\n")
        # print(str(j) + " target values written")
        i += 1
    file.close()
    print(str(i) + " data lines written")
    

# saveMatrix( numpy.array([[1,2,3],[2,4,2]]), numpy.array([[1,1],[2,2]]), ["a","b","c","class1","class2"], "AAAA")
# exit()

def getAll(file):
    print("Get all " + file)
    train = []
    trainExtra = []
    users = getUserData(file)
    i = 1
    for (user, (country, sex)) in users.items():
        if i%50 == 0:
            print("Get all: Processing " + str(i) + "/" + str(len(users)))
        i += 1
        tweets = getUserTweets(user,country)
        train_data = []
        for tweet in tweets:
            train.append(tweet)
            trainExtra.append([user,country,sex])

    train = numpy.array(train)
    trainExtra = numpy.array(trainExtra)
    print(train.shape)
    print(trainExtra.shape)
    return train, trainExtra


def predict(classIndex):
    # calculate matrix for training data
    X, y, features_training = createMatrix("training.txt", classIndex)
    saveMatrix(X, y, features_training, "training_"+SUPERCLASSES[classIndex])

    # calculate matrix for test data
    test_X, test_y, features_test = createMatrix("test.txt", classIndex)
    saveMatrix(X, y, features_test, "test_"+SUPERCLASSES[classIndex])

    trainAndPredict(X, y, test_X, test_y, classIndex)







exit()

classIndex = 0
predict(0)
predict(1)

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