# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:43:19 2016

@author: hitanshu.pande
"""
import pickle
import pandas as pd
# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

# Import the necessary methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

# Variables that contains the user credentials to access Twitter API 
ACCESS_TOKEN = '168198323-s4JusbpAbGJ4pVlpT1QiimtOmHhinjFRtpFSfXMS'
ACCESS_SECRET = 'VZgI6XG1Zq6v7I7KbwXuvTmEUuo3LM3Jc2ejWL6B83ZHe'
CONSUMER_KEY = 'RG5RoL6jZziKeKBn9JfdkEd5U'
CONSUMER_SECRET = 'dkBDJLjyjNMXPZx2kCgjtpmsTAwdbCa9NJPpHQqzwnP8qksCkJ'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Initiate the connection to Twitter Streaming API
twitter= Twitter(auth=oauth)

# Search for a keyword
twitter_data = twitter.search.tweets(q='#Trump', result_type='recent', lang='en', count=100)

#for result in twitter_data["statuses"]:
#	print ("(%s) @%s %s" % (result["created_at"], result["user"]["screen_name"], result["text"]))

#Create a data file of 100 tweets extracted from twitter
pickle.dump(twitter_data, open('twitter_data.p', 'wb'))

#Import in to pandas
df = pd.io.json.json_normalize(twitter_data['statuses'])
sample = df['text']


#Print all the tweet text:  df['text'] 

#Import training data from labelled tweet corpus
twitter_train = pd.read_csv('full-corpus.csv')
all_data = twitter_train


# This library transforms the training text data into a CountVectorizer. What it does is creates each 
# unique word as a column and assigns the frequency for each word per tweet in that row

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(all_data['TweetText'].values)
X_train_counts.shape

# Add a new column to all_data and assign the value 0,1,2,3 for the different sentiments
all_data['result'] = 0
all_data.loc[all_data['Sentiment'] == 'negative', 'result'] = 1
all_data['Sentiment'].unique()
all_data.loc[all_data['Sentiment'] == 'neutral', 'result'] = 2
all_data.loc[all_data['Sentiment'] == 'irrelevant', 'result'] = 3


# Split the dataset into train and test sets
from sklearn.cross_validation import StratifiedKFold
skf = StratifiedKFold(all_data['result'].values, 2, shuffle=True)

skf_list = list(skf)
skf_list[0]

skf_list[0][0].shape
skf_list[0][1].shape
skf_list[1][0].shape
skf_list[1][1].shape

train = skf_list[1][0]

skf_train = skf_list[1][0]
skf_test = skf_list[1][1]

#this is the training data with CountVector data, split into a fold
train = X_train_counts[skf_train]
train.shape
X_train_counts.shape
#this is the test data with CountVector data, split into a fold
test = X_train_counts[skf_test]
#train label contains the result of the tweet (0,1,2,3) for train data
train_label = all_data['result'].values[skf_train]
train_label.shape
#test label contains the result of the tweet (0,1,2,3) for test data
test_label = all_data['result'].values[skf_test]


#Implement the classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train, train_label)
pred = clf.predict(test)

#Plot the prediction values
import matplotlib.pyplot as plt
plt.hist(test_label, bins=4)
plt.hist(pred, bins=4)

#check the accuracy score
from sklearn.metrics import accuracy_score 
accuracy_score(test_label, pred)

#tweak the value of Alpha for the clf.fit
alpha_list = [0.01, 0.05, 0.1, 0.5, 1, 5, 10]


for alpha in alpha_list:
    print('alpha', alpha)
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train, train_label)
    pred = clf.predict(test)
    print('Score is:', accuracy_score(test_label, pred))
    

#Continue tweaking
for i in range(10):
    alpha = (i+1)*0.1
    print('alpha', alpha)
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train, train_label)
    pred = clf.predict(test)
    print('Score is:', accuracy_score(test_label, pred))

for i in range(10):
    alpha = 0.5 + i*0.01
    print('alpha', alpha)
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train, train_label)
    pred = clf.predict(test)
    print('Score is:', accuracy_score(test_label, pred))

for i in range(10):
    alpha = 0.4 + i*0.01
    print('alpha', alpha)
    clf = MultinomialNB(alpha=alpha)
    clf.fit(train, train_label)
    pred = clf.predict(test)
    print('Score is:', accuracy_score(test_label, pred))



#Final prediction
clf = MultinomialNB(alpha=0.5)
clf.fit(train, train_label)
pred = clf.predict(sample)