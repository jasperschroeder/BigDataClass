###############################################################################
# title:        00-get-features.py
# created on:   May 15, 2021
# summary:      run in the background
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
    if (i % 1000 == 0):
        print('Iteration ' + str(i))
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
