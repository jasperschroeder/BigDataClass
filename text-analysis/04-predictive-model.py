###############################################################################
# title:        04-predictive-model.py
# created on:   May 13, 2021
# summary:      using information from threads to predict high volatility
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# idea: can we use information from the threads and topics to predict high volatility?


###############################
# get 30-day rolling volatility: from Jasper's notebook :D
###############################

btc = pd.read_csv('bpi.csv')
Day = btc['Date']
def str_to_time(elem):
    day = datetime.datetime.strptime(elem, '%Y-%m-%d')
    return day

btc['Date'] = btc['Date'].apply(str_to_time)
btc = btc.set_index('Date')

ch_btc = [math.nan]
ch_btc_pct = [math.nan]
for i in range(1, len(btc['BPI'])):
    ch_btc.append(btc['BPI'].iloc[i]-btc['BPI'].iloc[i-1])
    ch_btc_pct.append((btc['BPI'].iloc[i]/btc['BPI'].iloc[i-1])-1)

vola = pd.DataFrame(ch_btc_pct).rolling(30).std()*np.sqrt(30)
vola.index = btc.index

########################
# define high volatility
########################

# high volatility: above 30%
def is_high(x):
    if math.isnan(x):
        return np.nan
    else:
        return (x > 0.3) + 0

vola['high'] = vola.apply(lambda x: (x > 0.3) + 0)
# what to do for the nan? for now leave it, we will just take it out of the modeling step
#vola.apply(lambda x: (x > 0.3)+0 if not math.isnan(x) else np.nan)

vola['high'].value_counts()
# 0    1618
# 1     209
# nice.

# ok, now we want to join the topic dataframe with this dataframe on date
# give the column 'Day' to vola
vola.index = range(len(vola))
vola['Day'] = Day
vola.columns = ['volatility', 'high', 'Day']

n_topic = 10
input_fname = 'nlp-data/df_topics_' + str(n_topic) + '.csv'
df = pd.read_csv(input_fname)
df.index = df['Unnamed: 0']
df.drop('Unnamed: 0', inplace = True, axis = 1)

merged_df = pd.merge(df, vola, on='Day')

# remove everything from 2020 onwards
merged_df = merged_df[merged_df['Day'] <'2020-01-01']

# remove the days where we have NaN volatility
merged_df = merged_df[~np.isnan(merged_df['volatility'])]

merged_df.shape # Out[125]: (143991, 21)

# can we use the features we got from the text to predict whether the thread was posted on
# a day with high volatility?

# question: do we have ordinal data?
# number of occurances could be ordinal if there is a ranking
# but here it's not a ranking per se
# so I guess we're ok

##############################
# model 1: logistic regression
##############################
# train/test split
# we have to use the past to predict the future, we cannot randomly sample from dates...
# on the other hand...I think we can, because date is not longer a factor

def special_division(n,d):
    return n/d if d > 0 else 0
vdiv = np.vectorize(special_division)

merged_df['words_per_sentence'] = vdiv(merged_df['word_count'],merged_df['n_sentences'])
merged_df['qmark_ratio'] = vdiv(merged_df['n_question_marks'],merged_df['word_count'])
merged_df['period_ratio'] = vdiv(merged_df['n_period'],merged_df['word_count'])
merged_df['exclam_ratio'] = vdiv(merged_df['n_exclam_marks'],merged_df['word_count'])


keep = ['comments', 'words_per_sentence', 'qmark_ratio', 'period_ratio',
       'exclam_ratio'] + ['topic_' + str(c) for c in range(n_topic)]

# keep = ['topic_' + str(c) for c in range(n_topic)]
# NOTE: even when we use the features we engineered...the topics are what make the difference!


X_full = merged_df[keep]
y_full = merged_df['high']

X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=420)

# look at the correlation of features in X_train

corr_mat = X_train.corr()

# Logistic Regression Model

model = LogisticRegression(solver='liblinear',
                           penalty='l1',
                           random_state=420)

model.fit(X_train, y_train)

model_1_preds = model.predict(X_test)

acc = np.mean(model_1_preds == y_test) # Out[151]: for 5 components 0.7525 nice..
print('Accuracy: ' + str(acc))

# # for 10 components it's 0.7526 so honestly we should keep 5

pd.DataFrame(np.transpose(model.coef_), index = X_train.keys())