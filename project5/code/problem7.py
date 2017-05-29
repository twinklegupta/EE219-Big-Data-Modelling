import json
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import urllib


class Reader:
    def __init__(self):
        self.sentences = []
        self.sentiments = []
        self.avg_sentiment = []
        self.sentences_list_windows = []
        for i in range(10):
            self.sentences_list_windows.append(list())

    def write_features(self, file_name):

        start_time_stamp = 1422831600  ##3:00 on feb 1, 2015

        end_time_stamp = 1422849600

        count = 0
        with open(file_name, 'r') as reader:
            for line in reader:
                tweet_json = json.loads(line)
                print(":::::::",count);
                count = count+1
                tweet_date = tweet_json["firstpost_date"]
                if tweet_date >= start_time_stamp and tweet_date <= end_time_stamp:

                    current_window = (tweet_date - start_time_stamp) / 1800
                    sentence = tweet_json["tweet"]["text"]
                    loc = tweet_json["tweet"]["user"]["location"]

                    print("process tweet : ", sentence)
                    sentence = sentence.replace("@", "")
                    sentence = sentence.replace("\n", " ")
                    sentence = re.sub(r'[^\x00-\x7F]+', ' ', sentence)
                    #b = TextBlob(sentence)
                    #if b.detect_language() != 'en':  # Read next tweet
                        #continue

                    sentence_splits = sentence.split(" ")
                    final_words = []
                    for split in sentence_splits:
                        if split.startswith("http") or split.startswith("#"):
                            pass
                        else:
                            final_words.append(split)
                    final_string = ""

                    for word in final_words:
                        final_string = final_string + " " + word
                    final_string.strip()
                    self.sentences_list_windows[current_window].append(final_string)
                    self.sentences.append(final_string)

        print ("Total number of Strings read :", len(self.sentences))

    def get_sentiments(self):
        f = open("seattle_sentiments.txt", 'w')

        for i in range(0, len(self.sentences_list_windows)):
            sentences_sentiments_window = []
            for sentence in self.sentences_list_windows[i]:
                if sentence == "":
                    continue
                #params = urllib.urlencode({'text': self.sentences[i]})
                #file_url = urllib.urlopen("http://text-processing.com/api/sentiment/", params)
                #data = file_url.read()
                #print (data)
                #try:
                #    response_json = json.loads(data)
                #    print ("Sentiment is " + response_json["label"])
                #    f.write(data + "\n")
                #except:
                #    pass

                analysis = TextBlob(sentence)
                # set sentiment
                f.write("sentiment :" + str(analysis.sentiment.polarity)+"\n")
                sentences_sentiments_window.append(analysis.sentiment.polarity)
            self.sentiments.append(sentences_sentiments_window)

        f.close()

    def read_file(self, name):
        pos_total = []
        neg_total = []

        values = []
        # with open("seattle_sentiments.txt", 'r') as reader:
        #     for line in reader:
        #         sentiment_json = json.loads(line)
        #         pos_value = sentiment_json["probability"]["pos"]
        #         neg_value = sentiment_json["probability"]["neg"]
        #         neutral_value = sentiment_json["probability"]["neutral"]
        #         pos_values.append(pos_value)
        #         neg_values.append(neg_value)
        #         neutral_values.append(neutral_value)
        #         if pos_value > neg_value:
        #             values.append(2)
        #         else:
        #             values.append(1)
        #         pos_total += sentiment_json["probability"]["pos"]
        #         neg_total += sentiment_json["probability"]["neg"]
        #         neutral_total += sentiment_json["probability"]["neutral"]
        for sentiment_window in self.sentiments:
            pos_total_val = 0
            neg_total_val = 0
            for val in sentiment_window:
                if(val > 0):
                    pos_total_val+=val
                else:
                    neg_total_val+=val
            pos_total.append(pos_total_val/len(sentiment_window))
            neg_total.append(-neg_total_val/len(sentiment_window))
            self.avg_sentiment.append((pos_total_val + neg_total_val)/len(sentiment_window))
            print("positive : ",pos_total_val)
            print("negative : ", neg_total_val)


        plt.plot(pos_total,'.r-')

        plt.plot(neg_total,'xb-')
        plt.plot(self.avg_sentiment, 'og-')
        #plt.show()
        img = name + ".png"
        plt.savefig(img)
        plt.close()
        #print ("Total Neutral", neutral_total)
        #plt.bar(range(0, len(values)), values, edgecolor='none', color='r')
        #plt.show()


r = Reader()
r.write_features("tweet_data/tweets_#gohawks.txt")
r.get_sentiments()
r.read_file("hawks")

r1 = Reader()
r1.write_features("tweet_data/tweets_#gopatriots.txt")
r1.get_sentiments()
r1.read_file("patriots")

plt.plot(r.avg_sentiment, 'og-')
plt.plot(r1.avg_sentiment,'xb-')
plt.savefig("bothTeams.png")
plt.close()