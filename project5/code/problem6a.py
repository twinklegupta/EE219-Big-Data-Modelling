import os
import json
import locale
import datetime
import importlib
import unicodedata
import matplotlib.pyplot as plt
Mas_File = "MATweets2.txt"
Wash_File = "WATweets2.txt"

tweets_superbowl = open("tweet_data/tweets_#superbowl.txt")
tweets_MA = open(Mas_File, "a")
tweets_WA = open(Wash_File, "a")

#Storing all the tweets from users with location in Washington or Massachussets
total_tweets = 0
WashCount = 0
MasCount = 0
WA_dict = dict()
MA_dict = dict()
WashUsers = set()
MasUsers = set()
with open("tweet_data/tweets_#superbowl.txt") as tweets:
    for tweet in tweets:
        #tweet = unicode(tweet, "utf-8")
        #parse JSON and fetch necessary data
        total_tweets = total_tweets + 1
        tweet_json = json.loads(tweet)
        location = tweet_json["tweet"]["user"]["location"]
        user_id = tweet_json["tweet"]["user"]["id"]
        tweet_content = tweet_json["tweet"]["text"]
        #check if the tweet is from WA or MA
        
        if ("Washington" in location or "WA" in location):
            if user_id not in WA_dict:
                WA_dict[user_id] = ""
            WA_dict[user_id] += " " + tweet_content 
            WashCount = WashCount + 1  
            WashUsers.add(user_id)
        elif("Massachusetts" in location or "MA" in location):
            if user_id not in MA_dict:
                MA_dict[user_id] = ""
            MA_dict[user_id] += " " + tweet_content 
            MasCount = MasCount + 1
            MasUsers.add(user_id)
            

print "Total Tweets: ", total_tweets
print "\nMassachusetts: \nUsers: ", len(MasUsers)
print "\nWashington: \nUsers: ", len(WashUsers)

for key in MA_dict:
    tweets_MA.write(MA_dict[key].encode('utf-8'))
    tweets_MA.write("$$DELIM$$")
for key in WA_dict:
    tweets_WA.write(WA_dict[key].encode('utf-8'))
    tweets_WA.write("$$DELIM$$")

            