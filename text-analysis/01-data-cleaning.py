###############################################################################
# title:        01-thread-info.py
# created on:   May 15, 2021
# summary:      feature engineering + text cleaning. outputs two datasets
###############################################################################

#######
# setup
#######

# run if needed:
# pip install https://github.com/andreasvc/readability/tarball/master

# import libraries
import pandas as pd
import numpy as np
import string
import spacy.cli
import matplotlib.pyplot as plt
import pdtext
import readability
from pdtext.tf import word_count
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from joblib import dump, load

# read in data
df = pd.read_csv('df_final.csv')

#####################
# remove missing text
#####################

# much of the text is missing; thus, we keep only the data with text
missing = (df["text"].isnull()) | ((df["text"] == '[deleted]')) | ((df["text"] == '[removed]'))
df = df.loc[~missing]

##############################
# clean the text in df for LDA
##############################
# stemming
eng_stemmer = SnowballStemmer("english")

def stem_sentence(sentence):
    new_sentence = ''
    for word in word_tokenize(sentence):
        new_sentence = new_sentence + ' ' + eng_stemmer.stem(word)
    return new_sentence.strip()

def clean_text(s, stem = True):
    # remove numbers: https://stackoverflow.com/questions/12851791/removing-numbers-from-string
    s = ''.join([i for i in s if not i.isdigit()])
    # remove punctuation:
    s = s.translate(str.maketrans('', '', string.punctuation))
    # stem
    if stem:
        s = stem_sentence(s)
    return(s)

# get clean text from which to extract topics
clean_text_df = df['text'].apply(clean_text)

clean_text_df = pd.DataFrame(clean_text_df, index=df.index)

# NOTE: a lot of the results are null. Get rid of them!
#is_null = clean_text_df.isnull()
#clean_text_df = clean_text_df.loc[~is_null]
# also do the same with the original df
#df = df.loc[~is_null]

clean_text_df.shape[0] == df.shape[0] # True, good

#############################
# export clean_text_df as csv
#############################
clean_text_df.to_csv('nlp-data/clean_text_df.csv', index=True)

###########
# sentiment
###########
# https://github.com/cjhutto/vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
fn_analyzer = lambda x: analyzer.polarity_scores(x)['compound']

# warning --slow
sentiment = df['text'].apply(fn_analyzer)

sentiment.head()

# export sentiment
sentiment.to_csv('nlp-data/sentiment.csv', index=True)


#####################
# feature engineering
#####################

# features we can get from the readability package
features = ['dale_chall', 'characters_per_word', 'syll_per_word', 'words_per_sentence', 'sentences_per_paragraph',
            'type_token_ratio', 'characters', 'syllables',
            'words', 'wordtypes', 'sentences', 'paragraphs',
            'long_words', 'complex_words', 'complex_words_dc',
            'tobeverb', 'auxverb', 'conjunction', 'pronoun',
            'preposition', 'nominalization']

# initiate empty features dataframe
feature_df = pd.DataFrame(np.nan, index = range(len(df)),
                          columns=features)

# function that outputs a row of features
def get_features(text, idx):
    results = readability.getmeasures(text, lang='en')
    DaleChall = pd.DataFrame(results['readability grades']['DaleChallIndex'],
                             columns = ['dale_chall'],
                             index = [idx])
    sentence_df = pd.DataFrame(results['sentence info'],
                               columns = results['sentence info'].keys(),
                               index = [idx])
    word_df = pd.DataFrame(results['word usage'],
                               columns = results['word usage'].keys(),
                               index = [idx])
    op = pd.concat([DaleChall, sentence_df, word_df], axis=1)
    return(op)



# fills the feature dataframe warning --slow
for i in range(len(df)):
    try:
        op = get_features(df['text'].iloc[i], idx=i)
    except:
        op = pd.DataFrame(np.nan, index = [i],columns=features)
    feature_df.iloc[i] = op

# add the index and id
# we need this to join the dataframes later on

feature_df.index = df.index
feature_df['id'] = df['id']

##########################
# export feature_df as csv
##########################
feature_df.to_csv('nlp-data/feature_df.csv', index=True)



