import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import readfile

# Library to scrape Google Play
from google_play_scraper import Sort, reviews
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import emoji



def preprocess_step(raw_text):
     # Remove HTML tags
    review_text = BeautifulSoup(raw_text).get_text()
    
    #Remove all numbers
    review_text = re.sub(r'\d+', ' ', review_text)
    
    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    
    # Remove punctuations
    text = re.sub(r'[\.\,\!\?\:\;\-\=]', ' ', letters_only)
    
    # Removing multiple spaces with single space
    space_text = re.sub(r'\s+', ' ', text)
    return space_text

def tolower_text(text):
    # Convert words to lower case and split each word up
    words = text.lower().split()
    return words


# Function to convert emojis to words
def convert_emoji_to_text(text):
    return emoji.demojize(text, delimiters=("",""))

def slangdict(text):
    slang = readfile.slangs()
    slang = slang.__dict__
    # performing split()
    temp = text.split()
    res = []
    
    for wrd in temp:

        # searching from lookp_dict
        res.append(slang.get(wrd, wrd))

    res = ' '.join(res)
    
    return("".join(str(res)))

def remove_stopwords(text):
    #lower the text
    words = tolower_text(text)
    
    # Searching through a set is faster than searching through a list 
    # Hence, we will convert stopwords to a set
    stops = set(stopwords.words('english'))
    
    # Adding on stopwords that were appearing frequently in both positive and negative reviews 
    stops.update(['app','shopee','shoppee','item','items','seller','sellers','bad']) 
    
    # Remove stopwords
    meaningful_words = [w for w in words if w not in stops]
    return("  ".join(meaningful_words))

# Write a function to convert raw text to a string of meaningful words
def stem_text(text):
        
    # Instantiate PorterStemmer
    p_stemmer = PorterStemmer()
    
    # Stem words
    meaningful_words = [p_stemmer.stem(w) for w in text]        
   
    text = "".join(meaningful_words)
    # Join words back into one string, with a space in between each word
    return(str(text))


def preprocess(reviews):
    # Pre-process the raw text
    reviews['content_stem'] = reviews['content'].map(preprocess_step)

    return reviews

def replaceslang(reviews):
    reviews['content_stem'] = reviews['content'].map(slangdict)
    return reviews

def replaceemoji(reviews):
    reviews['content_stem'] = reviews['content'].map(convert_emoji_to_text)
    return reviews

def replacestopwords(reviews):
    reviews['content_stem'] = reviews['content_stem'].map(remove_stopwords)
    return reviews

def replacestem_text(reviews):
    reviews['content_stem'] = reviews['content_stem'].map(stem_text)
    return reviews

def train(reviews):
   
    X = reviews[[cols for cols in reviews.columns if cols != 'target']]
    y = reviews['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Merge X_train and y_train back together using index
    train = pd.merge(X_train, y_train, left_index=True, right_index=True)

    # Merge X_test and y_test back together using index
    test = pd.merge(X_test, y_test, left_index=True, right_index=True)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train

def test(reviews):
   
    X = reviews[[cols for cols in reviews.columns if cols != 'target']]
    y = reviews['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Merge X_train and y_train back together using index
    train = pd.merge(X_train, y_train, left_index=True, right_index=True)

    # Merge X_test and y_test back together using index
    test = pd.merge(X_test, y_test, left_index=True, right_index=True)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return test




def vader(train):
    sia = SentimentIntensityAnalyzer()
    # Create an empty list to append the polarity scores
    sia_list = []

# Loop through the text in the training dataset and create a dictionary of the VADER scores
    for text in train['content']:
        sia_dict = {}
        sia_dict['neg'] = sia.polarity_scores(text)['neg'] # Negative score
        sia_dict['neu'] = sia.polarity_scores(text)['neu'] # Neutral score
        sia_dict['pos'] = sia.polarity_scores(text)['pos'] # Positive scores
        sia_dict['compound'] = sia.polarity_scores(text)['compound'] # Compound scores
        sia_list.append(sia_dict) # Append the dictionary of scores to the sia_lis
    
    # Create a dataframe from the sia_list
    sia_df = pd.DataFrame(sia_list)
    # Include 'content' and 'target' in sia_df
    sia_df['content'] = train['content']
    sia_df['target'] = train['target']

    return sia_df

    
