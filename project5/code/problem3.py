#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 12:51:42 2017

@author: twinklegupta
"""

import json
import datetime
from datetime import timedelta
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.figure as fig
import numpy as np


def extract_features(file):
    data = open("tweet_data/" + training[file])
    features = {}
    impressions =0 
    users = {}  # dictionnary of sets that stores unique users per hour

    for tweet in data:
        tweet_dict = json.loads(tweet)  # extract json data as dictionary
        totalFollowers = tweet_dict["author"]["followers"]
        retweets = tweet_dict["metrics"]["citations"]["total"]
        user = tweet_dict["tweet"]["user"]["id"]
        impressions += tweet_dict['metrics']['impressions']
        tweet_time = tweet_dict["firstpost_date"]
        favorite_count = tweet_dict["tweet"]["favorite_count"]  # number of 'favourites' the tweet received
        ranking_score = tweet_dict["metrics"]["ranking_score"]  # stores rank of tweet
        urls = len(tweet_dict["tweet"]["entities"]["urls"])  # stores if URLs are present in the tweet
        if urls > 0:
            urls = 1
        else:
            urls = 0

        tweet_time = datetime.datetime.fromtimestamp(tweet_time)
        new_tweet_time = datetime.datetime(tweet_time.year, tweet_time.month, tweet_time.day, tweet_time.hour, 0, 0)
        new_tweet_time = str(new_tweet_time)

        if new_tweet_time not in features:  # if that timestamp has not been seen before
            features[new_tweet_time] = {'totalTweets': 0, 'retweets': 0, 'time': -1, 'followers': 0, 
                                         'favorite_count': 0, 'ranking_score': 0, 'urls': 0,'user_count':0,
                                        'impressions' : 0
                                        }
            users[new_tweet_time] = set([])
        features[new_tweet_time]['totalTweets'] += 1
        features[new_tweet_time]['retweets'] += retweets
        features[new_tweet_time]['time'] = tweet_time.hour
        features[new_tweet_time]['favorite_count'] += favorite_count
        features[new_tweet_time]['ranking_score'] += ranking_score
        features[new_tweet_time]['urls'] += urls
        features[new_tweet_time]['impressions'] += impressions

        if user not in users[new_tweet_time]:
            users[new_tweet_time].add(user)
            features[new_tweet_time]['user_count'] += 1
            features[new_tweet_time]['followers'] += totalFollowers
 

    return features


# this function extracts the actual ground truth values of data and the data used for predicting the value for the next hour
def extract_labels(features_input):
    features = dict()
    for timeHour in features_input:
        current_time = datetime.datetime.strptime(timeHour, "%Y-%m-%d %H:%M:%S")
        features[current_time] = features_input[timeHour]
    train_data = []
    ground_truths = []

    start_time = min(features.keys())
    end_time = max(features.keys())

    current_time = start_time
    while current_time <= end_time:

        next_time = current_time + timedelta(hours=1)
        if next_time in features:  # if data for next hour is already available
            nextTotalTweets = features[next_time]['totalTweets']  # assign the known value to tweet count for next hour
        else:
            nextTotalTweets = 0  # initilaise tweet count for next hour to zero if data is not available

        if current_time in features:
            ground_truths.append([nextTotalTweets])  # the value to be predicted it the tweet count for next hour, so ground truth contains value for next hour
            train_data.append(features[current_time].values())  # treat data for current time as training data
        else:  # data for current time stamp is not available, assume all values to be zero
            temp =  {'totalTweets': 0, 'retweets': 0, 'time': -1, 'followers': 0, 
                                         'favorite_count': 0, 'ranking_score': 0, 'urls': 0,'user_count':0,
                                        'impressions' : 0
                                        }
            train_data.append(temp.values())
            ground_truths.append([nextTotalTweets])
        current_time = next_time
    return train_data, ground_truths


'''
9 features for this moedl are :- 
{
1)'totalTweets'
2)'retweets'
3)'time'
4)'followers'
5)'favorite_count'
6)'ranking_score'
7)'urls'
8)'user_count'
9)'impressions' 
}
'''

training = ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt",
            "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtags = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]
feature_names=['totalTweets','retweets','time','followers',
                                         'favorite_count','ranking_score', 'urls','user_count',
                                        'impressions' ]
best_features = [[1,4,7],[4,5,7],[2,3,6],[4,6,7],[2,3,9],[3,4,9]]

for i in range(len(training)):
    features = extract_features(i)
    train_data, ground_truths = extract_labels(features)
    train_data = sm.add_constant(train_data)
    regressionModel = sm.OLS(ground_truths, train_data)
    fitRegressionModel = regressionModel.fit()

    
#1:-1,4,7
#2 :- 4,7,5
#3 :- 2,3,6
#4 :- 4,6,7
#5 :- 2,3,9
#6 :- 3,4,9
#     if(i==0):
#         #Plotting and storing scatter plots for best three features
    for j in best_features[i]:
        print "best _Features "+str(i)+"--"+str(j-1)
        plt.gca().scatter(ground_truths,train_data[:,j],color='r')
        print "Lable _curent "+feature_names[j-1]
        xlable="Feature : "+feature_names[j-1]
        plt.xlabel(xlable)
        plt.ylabel('Tweets for next hour')
        plt.draw()
        imageName = hashtags[i] + "_best_feature_"+str(j)+"png"
        plt.savefig(imageName)
        plt.close()
        print(fitRegressionModel.summary())
        with open("linear_regression_problem_3_result"+hashtags[i]+".txt", 'wb') as fp:
            print >>fp, fitRegressionModel.summary()
