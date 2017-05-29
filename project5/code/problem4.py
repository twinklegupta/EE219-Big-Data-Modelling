import os
import json
import datetime
from sets import Set
from datetime import timedelta
import os.path
import statsmodels.api as sm
#import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
import numpy as np
import collections


# 'totalTweets': 0, 'retweets': 0, 'time': -1, 'followers': 0, 
#'favorite_count': 0, 'ranking_score': 0, 'urls': 0,'user_count':0,
#'impressions' : 0


def per_period_per_hour_features_extract(file_index):
    
    data= open("tweet_data/"+training[file_index],"r")
    
    start_date = datetime.datetime(2015,2,1,8,0,0)
    
    end_date = datetime.datetime(2015,2,1,20,0,0)
    
    per_hour_features = {}
    
    users_per_hour = {} 
    for tweet in data: 
        
        tweet_dict = json.loads(tweet) #Store an individual tweet as JSON object
        current_time = tweet_dict["firstpost_date"] #The time when the tweet was posted
        current_time = datetime.datetime.fromtimestamp(current_time) 
        converted_time = datetime.datetime(current_time.year, current_time.month, current_time.day, current_time.hour, 0, 0)
        converted_time = unicode(converted_time)
        current_user_id = tweet_dict["tweet"]["user"]["id"] 
        retweet_count = tweet_dict["metrics"]["citations"]["total"]
        followers_count = tweet_dict["author"]["followers"]
        impressions= tweet_dict['metrics']['impressions']
        url_count = len(tweet_dict["tweet"]["entities"]["urls"])
        if url_count>0:
            url_count = 1 
        else:
            url_count = 0
        favorite_count = tweet_dict["tweet"]["favorite_count"]
        ranking_score = tweet_dict["metrics"]["ranking_score"]
        if converted_time not in per_hour_features:
            per_hour_features[converted_time] =  {'totalTweets': 0, 'retweets': 0, 'time': -1, 'followers': 0, 
                                     'favorite_count': 0, 'ranking_score': 0, 'urls': 0,'user_count':0,
                                    'impressions' : 0
                                    }
            users_per_hour[converted_time] = Set([])
        per_hour_features[converted_time]['totalTweets'] += 1
        per_hour_features[converted_time]['retweets'] += retweet_count
        per_hour_features[converted_time]['time'] = current_time.hour
        per_hour_features[converted_time]['urls'] += url_count
        per_hour_features[converted_time]['favorite_count'] += favorite_count
        per_hour_features[converted_time]['ranking_score'] += ranking_score
        per_hour_features[converted_time]['impressions'] +=impressions
        if current_user_id not in users_per_hour[converted_time]:
            users_per_hour[converted_time].add(current_user_id)
            per_hour_features[converted_time]['followers'] += followers_count
            per_hour_features[converted_time]['user_count'] += 1
                
    modified_per_hour_features = {}
    for time_value in per_hour_features:
        cur_hour = datetime.datetime.strptime(time_value, "%Y-%m-%d %H:%M:%S")
        features = per_hour_features[time_value]
        modified_per_hour_features[cur_hour] = features    
    all_keys = modified_per_hour_features.keys()
    first_period_per_hour_features = {}   
    second_period_per_hour_features = {}
    third_period_per_hour_features = {}    
    for key in all_keys:
        if(key < start_date):
            first_period_per_hour_features[key] = modified_per_hour_features[key]
        elif(key >= start_date and key <= end_date):
            second_period_per_hour_features[key] = modified_per_hour_features[key]
        else:
            third_period_per_hour_features[key] = modified_per_hour_features[key]
    return modified_per_hour_features, first_period_per_hour_features, second_period_per_hour_features, third_period_per_hour_features
   

def variables_lables_matrix(per_hour_features):
    start_time = min(per_hour_features.keys()) 
    end_time = max(per_hour_features.keys())    
    train_data = []
    ground_truths = []
    current_time = start_time
    while current_time <= end_time: 
        
        next_time = current_time+timedelta(hours=1) 
        if next_time in per_hour_features: # if data for next hour is already available
            nextTotalTweets = per_hour_features[next_time]['totalTweets']  # assign the known value to tweet count for next hour
        else :
            nextTotalTweets = 0  # initilaise tweet count for next hour to zero if data is not available
        
        if current_time in per_hour_features: 
            ground_truths.append([nextTotalTweets]) # the value to be predicted it the tweet count for next hour, so ground truth contains value for next hour
            train_data.append(per_hour_features[current_time].values()) # treat data for current time as training data
        else: # data for current time stamp is not available, assume all values to be zero
            temp = {'totalTweets': 0, 'retweets': 0, 'time': -1, 'followers': 0, 
                                         'favorite_count': 0, 'ranking_score': 0, 'urls': 0,'user_count':0,
                                        'impressions' : 0
                                        }  
            train_data.append(temp.values())
            ground_truths.append([nextTotalTweets])
        current_time = next_time
    return train_data, ground_truths

def cross_validation(train_data,ground_truths):
    all_prediction_errors = []
    kf = KFold(len(train_data), n_folds=10)
    for train, test in kf:
        train_predictors = [train_data[i] for i in train]
        test_predictors = [train_data[i] for i in test]
        train_labels = [ground_truths[i] for i in train]
        test_labels = [ground_truths[i] for i in test]
        train_labels = sm.add_constant(train_labels)        
        model = sm.OLS(train_labels, train_predictors)
        results = model.fit()
        test_labels_predicted = results.predict(test_predictors)
        prediction_error = abs(test_labels_predicted - test_labels)
        prediction_error = np.mean(prediction_error)
        all_prediction_errors.append(prediction_error)
    return np.mean(all_prediction_errors)        



def perform_classification(X, Y):
    """
    :return: 10 fold cross validation of X and Y
    """

    average_error = []
    kf = KFold(len(X), n_folds=10, shuffle=False, random_state=None)

    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        y_train = collections.deque(y_train)
        y_train.rotate(-1)
        y_train = list(y_train)
        X_train = list(X_train)

        result = sm.OLS(y_train, X_train).fit()
        test_prediction = result.predict(X_test)
        average_error.append(np.mean(abs(test_prediction - y_test)))
        return np.mean(average_error)   
        
folder_name = "tweet_data"
training =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtags = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]

periods=[1,2,3]
period_name=['first_period_per_hour_features','second_period_per_hour_features','third_period_per_hour_features'];

for i in range(len(training)):
    modified_per_hour_features, first_period_per_hour_features, second_period_per_hour_features, third_period_per_hour_features = per_period_per_hour_features_extract(i)
    train_data, ground_truths = variables_lables_matrix(modified_per_hour_features)
    period_name=[first_period_per_hour_features,second_period_per_hour_features,third_period_per_hour_features];

    #average_cv_pred_error = cross_validation(train_data,ground_truths)
    average_cv_pred_error = perform_classification(np.array(train_data),np.array(ground_truths))
    print "The average prediction error for full cross-validation of", hashtags[i], " is ", average_cv_pred_error
   
    for j in periods:
        train_data, ground_truths = variables_lables_matrix(period_name[j-1])
        average_cv_pred_error =perform_classification(np.array(train_data),np.array(ground_truths))
        print "The average prediction error using cross-validation for Period "+str(j), hashtags[i], " is ", average_cv_pred_error
        


# 
# The average prediction error for full cross-validation of #gohawks  is  7.23854571533
# The average prediction error using cross-validation for Period 1 #gohawks  is  6.66358764162
# The average prediction error using cross-validation for Period 2 #gohawks  is  26930.592272
# The average prediction error using cross-validation for Period 3 #gohawks  is  3921.00642844
# The average prediction error for full cross-validation of #gopatriots  is  0.662027684759
# The average prediction error using cross-validation for Period 1 #gopatriots  is  0.373799273238
# The average prediction error using cross-validation for Period 2 #gopatriots  is  671.134110061
# The average prediction error using cross-validation for Period 3 #gopatriots  is  17.2043735787
# The average prediction error for full cross-validation of #nfl  is  2.79884372525
# The average prediction error using cross-validation for Period 1 #nfl  is  3.32135896744
# The average prediction error using cross-validation for Period 2 #nfl  is  7496.3000768
# The average prediction error using cross-validation for Period 3 #nfl  is  166.987414963
# The average prediction error for full cross-validation of #patriots  is  13.3817974166
# The average prediction error using cross-validation for Period 1 #patriots  is  2.74245505624
# The average prediction error using cross-validation for Period 2 #patriots  is  64506.8881789
# The average prediction error using cross-validation for Period 3 #patriots  is  492.26330596
