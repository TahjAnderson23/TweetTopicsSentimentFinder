# This code was created by Skanda Vivek and shared on medium

import sys
import os
from ctt import clean
import pickle
import re
from matplotlib.style import available
import tweepy
from tweepy import OAuthHandler
#import twitter
from textblob import TextBlob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#from IPython.display import clear_output
import matplotlib.pyplot as plt
import configparser
import matplotlib.pyplot as plt

import json
#%matplotlib inline

#Authenticate 
client = tweepy.Client("")
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

auth = tweepy.AppAuthHandler(api_key, api_key_secret)
#auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)


if (not api):
    print ("Can’t Authenticate")
    sys.exit(-1)

def pull_by_location():
    tweet_lst=[]
    geoc="33.7490,-84.3880,40mi"
    for tweet in tweepy.Cursor(api.search_tweets,q="Atlanta", geocode=geoc).items(1000):
        tweetDate = tweet.created_at.date()
        tweetDate = datetime(2022,2,10)
        if(tweet.coordinates != None):
            print('FOUND TWEET!')
            tweet_lst.append([tweetDate,tweet.id,tweet.
                    coordinates['coordinates'][0],
                    tweet.coordinates['coordinates'][1],
                    tweet.user.screen_name,
                    tweet.user.name, tweet.text,
                    tweet.user._json['geo_enabled']])
            #print('FOUND TWEET!')
    tweet_df = pd.DataFrame(tweet_lst, columns=['tweet_dt', 'id', 'lat','long','username', 'name', 'tweet','geo'])

    #pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(tweet_df)
    print(tweet_df.describe())
    #tweet_df.to_csv('Atlanta_tweets.csv')
    plt.title('Geotagged tweets in Atlanta')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.scatter(x=tweet_df['long'], y=tweet_df['lat'])
    plt.show()


def pull_by_hashtag():
    tweet_lst2=[]
    for tweet in tweepy.Cursor(api.search_tweets, q='Climate Change -filter:retweets').items(1000):
        if tweet.lang == "en":
            tweetDate = tweet.created_at.date()
            #tweetDate = datetime(2022,2,13)
            tweet_lst2.append([tweetDate,tweet.id,
                    tweet.user.screen_name,
                    tweet.user.name, tweet.text,
                    ])
        
    tweet_df2 = pd.DataFrame(tweet_lst2, columns=['tweet_dt', 'id','username', 'name', 'tweet'])
    print(tweet_df2)
    find_sentiment(tweet_df2)

    #print(tweet_df2.describe())
    #tweet_df2.to_csv('superbowl_halftime_tweets.csv')

def find_trending_tweets():
    available_loc = api.available_trends()

    df = pd.DataFrame.from_records(available_loc)
    print(df)

def plot_data():
   
    data = pd.read_csv('Atlanta_tweets.csv')

    plt.scatter(x=data['long'], y=data['lat'])
    plt.show()


def clean_tweets(txt):
    hashtagPattern = re.compile("#\w+")
    mentionPattern = re.compile("@\w+")
    txt = re.sub(hashtagPattern, "", str(txt))
    txt = re.sub(mentionPattern, "", str(txt))
    
    txt = " ".join(txt.split())
    
    return txt

def find_sentiment(df):
    vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
    classifier = pickle.load(open('models/SVCclassifier.sav', 'rb'))

    df['tweet'] = df['tweet'].apply(clean_tweets)

    cleanedText = []
    for text in df["tweet"]:
    # preprocess to remove unwanted
        text = clean.kitchen_sink(text)
        cleanedText.append(text)
    
    df.loc[:, "tweet"] = cleanedText
    #print(df.head())
    sentiment_df = pd.DataFrame(columns = ["text", "sentiment"])
    sentiment_arrays = []
    for text in df["tweet"]:
        #sentiment_df["text"] = text
        text_vector = vectorizer.transform([text])
        result = classifier.predict(text_vector)
        #sentiment_df["sentiment"] = result
        sentiment_arrays.append(result)
    
    sen_df = pd.DataFrame(sum(map(list, sentiment_arrays), []))
    df["sentiment"] = sen_df[0]
    print(df)



#plot_data()
pull_by_hashtag()
#find_trending_tweets()