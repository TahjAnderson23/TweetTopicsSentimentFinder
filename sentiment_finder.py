# utilities
import re
import numpy as np
import pandas as pd
import nltk
# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing

#Sentiment finder
from textblob import TextBlob
# Importing the dataset
##DATASET_COLUMNS=['ids','date','user','text']
#DATASET_ENCODING = "ISO-8859-1"
df = pd.read_csv('cleaned_superbowl_halftime_tweets.csv')


df.tweet = df.tweet.astype(str) 
df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)

df['polarity'] = df.apply(lambda x: TextBlob(x['tweet']).sentiment.polarity, axis=1)
df['tweet']=df['tweet'].str.lower()

tokenizer = RegexpTokenizer(r'\w+')
df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
#print(df['tweet'].head())

st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data

df['tweet']= df['tweet'].apply(lambda x: stemming_on_text(x))
print(df['tweet'].head())

lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data

df['tweet'] = df['tweet'].apply(lambda x: lemmatizer_on_text(x))
print(df['tweet'].head())

tweet_list = df['tweet']
df["tweet"] = df["tweet"].astype(str)

X=df.tweet
y=df.polarity

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.05, random_state =26105111)
print(X_train.dtype)
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)

vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)

def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


print("X TRAIN DTYPE",X_train.dtype)
print("Y TRAIN DTYPE",y_train.dtype)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, training_scores_encoded)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)