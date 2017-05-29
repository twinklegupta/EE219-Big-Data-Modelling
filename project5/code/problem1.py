#!/usr/bin/env python3
# encoding=utf8
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:10:30 2017

@author: twinklegupta

"""




import os
import json
import datetime
#from sets import Set
from datetime import timedelta
import matplotlib.pyplot as plt
import sys
import importlib
# reload(sys)
# sys.setdefaultencoding("utf-8")
# importlib.reload (sys)  


                   

# calculates the asked statistics for every tweet for a particular hashtag
def calculate_values(file_index):
    data= open("tweet_data/"+training[file_index],"r")
#     data= data.decode("utf8")

    
    totalFollowers = 0.0
    retweets = 0.0
    totalTweets = 0.0
    users = set([])
    start_time = datetime.datetime(2017,1,1)
    end_time = datetime.datetime(2000,1,1)
    hour_dict = {}
    for tweet in data:
        totalTweets = totalTweets + 1
        tweet_dict = json.loads(tweet)
        user_id = tweet_dict["tweet"]["user"]["id"]
        if user_id not in users:
            users.add(user_id)
            totalFollowers += tweet_dict["author"]["followers"]
        retweets += tweet_dict["metrics"]["citations"]["total"]
        tweet_time = tweet_dict["firstpost_date"]
        tweet_time = datetime.datetime.fromtimestamp(tweet_time)
        if tweet_time < start_time:
            start_time = tweet_time
        if tweet_time > end_time:
            end_time = tweet_time
        keyTime = str(datetime.datetime(tweet_time.year,tweet_time.month, tweet_time.day, tweet_time.hour, 0, 0))
        if keyTime not in hour_dict:
            hour_dict[keyTime] = 0.0
        hour_dict[keyTime] += 1.0;
        
    activeTime = int((end_time - start_time).total_seconds()/3600 + 0.5)   
    return totalFollowers, totalTweets,retweets, activeTime, len(users), hour_dict
        

#plot histogram to show number of tweets per hour 
def  plot_hist(hour_dict,file):
    new_hour_dict = dict()
    for timeHour in hour_dict:
        current_time = datetime.datetime.strptime(timeHour, "%Y-%m-%d %H:%M:%S")
        new_hour_dict[current_time] = hour_dict[timeHour]   
    initial_time = min(new_hour_dict.keys())
    final_time = max(new_hour_dict.keys())
    toPlot= []
    current_time = initial_time
    while current_time <= final_time:
        if current_time in new_hour_dict:
            toPlot.append(new_hour_dict[current_time])
        else:
            toPlot.append(0)
        current_time += timedelta(hours=1)        
    
    #Plotting the histogram
    plt.figure(figsize=(20, 8))
    plt.title("Number of Tweets per Hour for " + hashtags[file])
    plt.ylabel("Number of Tweets")
    plt.xlabel("Hours")
    plt.bar(range(len(toPlot)), toPlot)
    plt.show()
 
    
training =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtags = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]


#read all files and calculate statistics
for i in range(len(training)):
    totalFollowers, totalTweets, retweets, activeTime, totalUsers, hour_dict = calculate_values(i)
    print ("Average number of tweets per hour for",  hashtags[i], "are", totalTweets/activeTime)
    print ("Average number of followers of users posting tweets for",  hashtags[i], "are", totalFollowers/totalUsers)
    print ("Average number of retweets for", hashtags[i], "are", retweets/totalTweets)
    if i == 2 or i == 5:
        plot_hist(hour_dict,i)