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

'''
This function takes the entire hash tag file as the input and calculates all the features
'''
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

def retrieve_hourly_features(file_index):
        
    data= open(file_index,"r")
    
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
    return modified_per_hour_features

 
'''
This function takes in the hourwise features for each hashtag as the input and computes the Predictor and Label Matrix
The ground_truth are the tweet counts for the next hour
'''
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
        
folder_name = "tweet_data"
training =  ["tweets_#gohawks.txt", "tweets_#gopatriots.txt", "tweets_#nfl.txt", "tweets_#patriots.txt", "tweets_#sb49.txt", "tweets_#superbowl.txt"]
hashtag_list = ["#gohawks", "#gopatriots", "#nfl", "#patriots", "#sb49", "#superbowl"]
#test_files_1 = ["sample1_period1.txt","sample4_period1.txt","sample5_period1.txt","sample8_period1.txt"]
#test_files_2 = ["sample2_period2.txt","sample6_period2.txt","sample9_period2.txt"]
#test_files_3 = ["sample3_period3.txt","sample7_period3.txt","sample10_period3.txt"]
test_files = ["sample1_period1.txt","sample2_period2.txt","sample3_period3.txt","sample4_period1.txt","sample5_period1.txt","sample6_period2.txt","sample7_period3.txt","sample8_period1.txt","sample9_period2.txt","sample10_period3.txt"]

x,first_period_per_hour_features_sb, second_period_per_hour_features_sb, third_period_per_hour_features_sb = per_period_per_hour_features_extract(5)
x,first_period_per_hour_features_nfl, second_period_per_hour_features_nfl, third_period_per_hour_features_nfl = per_period_per_hour_features_extract(3)     
train_data1_sb, ground_truth1_sb = variables_lables_matrix(first_period_per_hour_features_sb)
train_data1_sb = sm.add_constant(train_data1_sb)      
train_data2_sb, ground_truth2_sb = variables_lables_matrix(second_period_per_hour_features_sb)
train_data2_sb = sm.add_constant(train_data2_sb)      
train_data3_sb, ground_truth3_sb = variables_lables_matrix(third_period_per_hour_features_sb)
train_data3_sb = sm.add_constant(train_data3_sb) 
train_data1_nfl, ground_truth1_nfl = variables_lables_matrix(first_period_per_hour_features_nfl)
train_data1_nfl = sm.add_constant(train_data1_nfl)      
train_data2_nfl, ground_truth2_nfl = variables_lables_matrix(second_period_per_hour_features_nfl)
train_data2_nfl = sm.add_constant(train_data2_nfl)      
train_data3_nfl, ground_truth3_nfl = variables_lables_matrix(third_period_per_hour_features_nfl)
train_data3_nfl = sm.add_constant(train_data3_nfl) 
model_1_sb = sm.OLS(ground_truth1_sb, train_data1_sb)
results_1_sb = model_1_sb.fit()
model_2_sb = sm.OLS(ground_truth2_sb, train_data2_sb)
results_2_sb = model_2_sb.fit()
model_3_sb = sm.OLS(ground_truth3_sb, train_data3_sb)
results_3_sb = model_3_sb.fit()
model_1_nfl = sm.OLS(ground_truth1_nfl, train_data1_nfl)
results_1_nfl = model_1_nfl.fit()
model_2_nfl = sm.OLS(ground_truth2_nfl, train_data2_nfl)
results_2_nfl = model_2_nfl.fit()
model_3_nfl = sm.OLS(ground_truth3_nfl, train_data3_nfl)
results_3_nfl = model_3_nfl.fit()

result_list=[results_1_sb,results_2_sb,results_3_sb,results_1_nfl,results_1_nfl,results_2_sb,results_3_nfl,results_2_sb,results_2_sb,results_3_nfl]

test_error = [0 for i in range(10)] 
for i in range(len(test_files)):
    #The sample1_period1 file has maximum #superbowl tags, so we use the corresponding model
    file_name = os.path.join("test_data 2", test_files[i])
    modified_hourwise_features = retrieve_hourly_features(file_name)    
    train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
    train_data = np.asarray(train_data)
    ground_truth = np.asarray(ground_truth)
    train_data = sm.add_constant(train_data)        
    test_ground_truth_predicted = result_list[i].predict(train_data)
    print "##$$",test_ground_truth_predicted
    test_error[i] = np.mean(abs(test_ground_truth_predicted - ground_truth))
#     if(i==1):
#         #The sample2_period2 file has maximum #superbowl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)
#         train_data = sm.add_constant(train_data)        
#         test_ground_truth_predicted_2 = results_2_sb.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_2 - ground_truth))      
#     if(i==2):
#         #The sample3_period3 file has maximum #superbowl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)        
#         train_data = sm.add_constant(train_data)        
#         test_ground_truth_predicted_3 = results_3_sb.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_3 - ground_truth)) 
#     if(i==3):
#         #The sample4_period1 file has maximum #nfl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)        
#         train_data = sm.add_constant(train_data)        
#         test_ground_truth_predicted_4 = results_1_nfl.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_4 - ground_truth))         
#     if(i==4):
#         #The sample5_period1 file has maximum #nfl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)        
#         train_data = sm.add_constant(train_data)        
#         test_ground_truth_predicted_5 = results_1_nfl.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_5 - ground_truth))         
#     if(i==5):
#         #The sample6_period2 file has maximum #superbowl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)        
#         train_data = sm.add_constant(train_data)        
#         test_ground_truth_predicted_6 = results_2_sb.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_6 - ground_truth))         
#     if(i==6):
#         #The sample7_period3 file has maximum #nfl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)        
#         train_data = sm.add_constant(train_data)              
#         test_ground_truth_predicted_7 = results_3_nfl.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_7 - ground_truth))         
#     if(i==7):
#         #The sample8_period1 file has maximum #nfl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)        
#         train_data = np.insert(train_data, [0], [[1],[1],[1],[1],[1]], axis=1)     
#         test_ground_truth_predicted_8 = results_2_sb.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_8 - ground_truth))     
#     if(i==8):
#         #The sample9_period2 file has maximum #superbowl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = sm.add_constant(train_data)        
#         test_ground_truth_predicted_9 = results_2_sb.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_9 - ground_truth)) 
#     if(i==9):
#         i=9
#         #The sample10_period3 file has maximum #nfl tags, so we use the corresponding model
#         file_name = os.path.join("test_data 2", test_files[i])
#         modified_hourwise_features = retrieve_hourly_features(file_name)    
#         train_data, ground_truth = variables_lables_matrix(modified_hourwise_features)
#         train_data = np.asarray(train_data)
#         ground_truth = np.asarray(ground_truth)        
#         train_data = np.insert(train_data, [0], [[1],[1],[1],[1],[1],[1]], axis=1)     
#         test_ground_truth_predicted_10 = results_3_nfl.predict(train_data)
#         test_error[i] = np.mean(abs(test_ground_truth_predicted_10 - ground_truth)) 

c=0
for i in test_error:
    print "error val for "+str(c)+"-->"+str(i)
    c+=1
