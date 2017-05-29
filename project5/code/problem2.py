#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:25:58 2017

@author: twinklegupta
"""

##############################################################################
#                                                                            #
#                                 Problem 2                                  #
#                                                                            #
##############################################################################     

import os
import json
import datetime
import statsmodels.api as sm
import os.path
from datetime import timedelta

# the function extracts the five mentioned features for the hashtag stored in "file"
def extract_features(file):
    data = open("tweet_data/"+training[file])
    features = {}
    users = {} #dictionnary of sets that stores unique users per hour
    
    for tweet in data:
        tweet_dict = json.loads(tweet) #extract json data as dictionary
        totalFollowers = tweet_dict["author"]["followers"]
        retweets = tweet_dict["metrics"]["citations"]["total"]
        user = tweet_dict["tweet"]["user"]["id"] 
        tweet_time = tweet_dict["firstpost_date"] 
        tweet_time = datetime.datetime.fromtimestamp(tweet_time) 
        new_tweet_time = datetime.datetime(tweet_time.year, tweet_time.month, tweet_time.day, tweet_time.hour, 0, 0)
        new_tweet_time = str(new_tweet_time)
            
        if new_tweet_time not in features: #if that timestamp has not been seen before
            features[new_tweet_time] = {'totalTweets':0, 'retweets':0,'time':-1, 'followers':0, 'max_followers':0}
            users[new_tweet_time] = set([])
        features[new_tweet_time]['totalTweets'] += 1
        features[new_tweet_time]['retweets'] += retweets
        features[new_tweet_time]['time'] = tweet_time.hour
        if user not in users[new_tweet_time]: 
            users[new_tweet_time].add(user)
            features[new_tweet_time]['followers'] += totalFollowers
            if totalFollowers > features[new_tweet_time]['max_followers']:
                features[new_tweet_time]['max_followers'] = totalFollowers
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
        
        next_time = current_time+timedelta(hours=1) 
        if next_time in features: # if data for next hour is already available
            nextTotalTweets = features[next_time]['totalTweets']  # assign the known value to tweet count for next hour
        else :
            nextTotalTweets = 0  # initilaise tweet count for next hour to zero if data is not available
        
        if current_time in features: 
            ground_truths.append([nextTotalTweets]) # the value to be predicted it the tweet count for next hour, so ground truth contains value for next hour
            train_data.append(features[current_time].values()) # treat data for current time as training data
        else: # data for current time stamp is not available, assume all values to be zero
            temp = {'totalTweets':0, 'retweets':0, 'followers':0, 'max_followers':0, 'time':current_time.hour}    
            train_data.append(temp.values())
            ground_truths.append([nextTotalTweets])
        current_time = next_time
    return train_data, ground_truths
   
training =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtags = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]


for i in range(len(training)):
    features = extract_features(i)
    print(features.values())
    train_data, ground_truths = extract_labels(features)
    #print(train_data)
    train_data = sm.add_constant(train_data)
    regressionModel = sm.OLS(ground_truths, train_data)
    fitRegressionModel = regressionModel.fit()
    print ( "Details for Linear Regression Model for " + hashtags[i] , fitRegressionModel.summary())