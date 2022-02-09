# This code was created by Skanda Vivek and shared on medium

import sys
import os
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#from IPython.display import clear_output
import matplotlib.pyplot as plt
import configparser
#%matplotlib inline

#Authenticate 

config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

auth = tweepy.AppAuthHandler(api_key, api_key_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

if (not api):
    print ("Can’t Authenticate")
    sys.exit(-1)


tweet_lst=[]
geoc="33.7490,-84.3880,1mi"
for tweet in tweepy.Cursor(api.search_tweets,q="food",geocode=geoc).items(1000):
    tweetDate = tweet.created_at.date()
    if(tweet.coordinates !=None):
        tweet_lst.append([tweetDate,tweet.id,tweet.
                coordinates['coordinates'][0],
                tweet.coordinates['coordinates'][1],
                tweet.user.screen_name,
                tweet.user.name, tweet.text,
                tweet.user._json['geo_enabled']])
tweet_df = pd.DataFrame(tweet_lst, columns=['tweet_dt', 'id', 'lat','long','username', 'name', 'tweet','geo'])

#pd.set_option("display.max_rows", None, "display.max_columns", None)
print(tweet_df)
