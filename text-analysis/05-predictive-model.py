###############################################################################
# title:        04-predictive-model.py
# created on:   May 13, 2021
# summary:      using information from threads to predict high volatility
###############################################################################
import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# idea: can we use information from the threads and topics to predict
# # whether the thread was posted when BTC had high volatility?

###############################
# get 30-day rolling volatility
# from Jasper's notebook :D
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
# def is_high(x):
#    if math.isnan(x):
#        return np.nan
#    else:
#        return (x > 0.3) + 0

vola['high'] = vola.apply(lambda x: (x > 0.3) + 0)
# what to do for the nan? for now leave it, we will just take it out of the modeling step
#vola.apply(lambda x: (x > 0.3)+0 if not math.isnan(x) else np.nan)

vola['high'].value_counts()
# 0    1618
# 1     209
# nice.

####################################
# Get the Dataset for Classification
####################################

# ok, now we want to join the topic dataframe with this dataframe on date
# give the column 'Day' to vola
vola.index = range(len(vola))
vola['Day'] = Day
vola.columns = ['volatility', 'high', 'Day']

# the threads
df = pd.read_csv('df_final.csv')
# again we remove the null values of df
missing = (df["text"].isnull()) | ((df["text"] == '[deleted]')) | ((df["text"] == '[removed]'))
df = df.loc[~missing]

feature_df = pd.read_csv('nlp-data/feature_df.csv', index_col = 0)
sentiment = pd.read_csv('nlp-data/sentiment.csv', index_col = 0)
tsne_df = pd.read_csv('nlp-data/df_topics_5.csv', index_col = 0)

# select some columns of the tsne_df
tsne_df = tsne_df[['id', 'topic_0', 'topic_1', 'topic_2',
       'topic_3', 'topic_4']]
dominant_topic = tsne_df[['topic_' + str(c) for c in range(5)]].idxmax(axis=1)

# add sentiment to the df
df['sentiment'] = sentiment

# merge
full_df_1 = pd.merge(df, feature_df, on='id')
full_df = pd.merge(full_df_1, tsne_df, on = 'id')

# select the right columns to keep
keep = ['Day', 'author', 'comments', 'sentiment', 'dale_chall',
       'type_token_ratio', 'characters', 'syllables', 'words', 'wordtypes',
       'sentences', 'paragraphs', 'long_words', 'complex_words',
       'complex_words_dc', 'tobeverb', 'auxverb', 'conjunction', 'pronoun',
       'preposition', 'nominalization', 'topic_0', 'topic_1', 'topic_2',
        'topic_3', 'topic_4']

full_df = full_df[keep]

merged_df = pd.merge(vola, full_df, on = 'Day')

# remove everything from 2020 onwards
merged_df = merged_df[merged_df['Day'] <'2020-01-01']

# remove the days where we have NaN volatility
merged_df = merged_df[~np.isnan(merged_df['volatility'])]

merged_df.shape # Out[125]: (143991, 28)
full_df = merged_df[['comments', 'sentiment',
       'dale_chall', 'type_token_ratio', 'characters', 'syllables', 'words',
       'wordtypes', 'sentences', 'paragraphs', 'long_words', 'complex_words',
       'complex_words_dc', 'tobeverb', 'auxverb', 'conjunction', 'pronoun',
       'preposition', 'nominalization', 'topic_0', 'topic_1', 'topic_2',
       'topic_3', 'topic_4']]

##################
# Correlation Heatmap
##################
mask = np.tril(full_df.corr())
sns.heatmap(full_df.corr(), annot=False, mask=mask)
plt.show()

# unsurprisingly, the engineered features are HIGHLY correlated.
# solution? PCA

#####
# PCA
#####
X = full_df[['comments', 'sentiment', 'dale_chall', 'type_token_ratio',
       'characters', 'syllables', 'words', 'wordtypes', 'sentences',
       'paragraphs', 'long_words', 'complex_words', 'complex_words_dc',
       'tobeverb', 'auxverb', 'conjunction', 'pronoun', 'preposition',
       'nominalization']]
# NOTE: there are some NAN in X. We will do nearest neighbor interpolation
# documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate
X = X.interpolate(method="nearest")

# Create a train / test set
#sentiment = sentiment.reset_index(drop=True)
#sentiment_binary = (sentiment > 0) + 0
#sentiment_binary = np.ravel(sentiment_binary)
high = merged_df['high']
# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    high,
                                                    test_size=0.2,
                                                    random_state=420)

# Fit PCA on X_train and X_test
# Scale
scaler = StandardScaler()
Z_train = scaler.fit_transform(X_train)
Z_test = scaler.fit_transform(X_test)

# Fit PCA to the Train set
pca = PCA(n_components=Z_train.shape[1], svd_solver='full')
pca.fit(Z_train)
# Transform
X_train_pca = pca.transform(Z_train)
X_test_pca = pca.transform(Z_test)

# determine variance explained
print(pca.explained_variance_ratio_)
plt.plot(range(Z_train.shape[1]), pca.explained_variance_ratio_)
plt.show()
plt.plot(range(Z_train.shape[1]), np.cumsum(pca.explained_variance_ratio_))
plt.show()
np.cumsum(pca.explained_variance_ratio_)
# 6 components explains 90% of the variance.
# so, let's take the 6 components and do PCA regression

# get the components
pca = PCA(n_components=6, svd_solver='full')
pca.fit(Z_train)
# Transform
X_train_pca = pca.transform(Z_train)
X_test_pca = pca.transform(Z_test)


# regression: sentiment vs PC's
npc = np.array(range(6)) + 1
pcnames = ['PC_' + str(i) for i in npc ]
X_train_pca = pd.DataFrame(X_train_pca, columns=pcnames)
X_test_pca = pd.DataFrame(X_test_pca, columns=pcnames)

#####################
# Logistic Regression
#####################

#-------------------------------------
# Model 1: Vanilla Logistic Regression
#-------------------------------------
model1 = LogisticRegression(solver='liblinear',
                           penalty='l1',
                           random_state=20)#,
                           #class_weight='balanced')
# fit
model1.fit(X_train_pca, y_train)
# predict
model1_preds = model1.predict(X_test_pca)

# accuracy
acc = np.mean(model1_preds == y_test)
print('Accuracy: ' + str(acc))
pd.DataFrame(np.transpose(model1.coef_), index = X_train_pca.keys())

# Evaluate the model's performance
cm = confusion_matrix(y_test, model1_preds,  normalize = "true")
sns.heatmap(cm, annot=True, cmap="Greens", fmt='g')
plt.show()
np.mean(model1_preds)

print(classification_report(y_test, model1_preds))

#--------------------------------------------
# Model 2: Class Weighted Logistic Regression
#--------------------------------------------
model2 = LogisticRegression(solver='liblinear',
                           penalty='l1',
                           random_state=20,
                           class_weight={0: 1.33, 1: 4.027})#class_weight='balanced')
# fit
model2.fit(X_train_pca, y_train)
# predict
model2_preds = model2.predict(X_test_pca)

# accuracy
acc = np.mean(model2_preds == y_test)
print('Accuracy: ' + str(acc))
pd.DataFrame(np.transpose(model2.coef_), index = X_train_pca.keys())

# Evaluate the model's performance
cm = confusion_matrix(y_test, model2_preds,  normalize = "true")
sns.heatmap(cm, annot=True, cmap="Greens", fmt='g')
plt.show()
np.mean(model2_preds)

print(classification_report(y_test, model2_preds))


# now let's use just the topics
#-----------------------------------------------
# Model 3: Logistic Regression with Topic Scores
#-----------------------------------------------
df = full_df[['topic_0', 'topic_1', 'topic_2',
        'topic_3', 'topic_4']]

df = df.reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(df,
                               high, test_size=0.2,random_state=420)

model3 = LogisticRegression(solver='liblinear',
                           penalty='l1',
                           random_state=420)

model3.fit(X_train, y_train)

model3_preds = model3.predict(X_test)

acc_train = np.mean(model3.predict(X_train) == y_train)
acc = np.mean(model3_preds == y_test) # Out[151]: for 5 components 0.7533 nice..
print('Accuracy: ' + str(acc))

pd.DataFrame(np.transpose(model3.coef_), index = X_train.keys())

# Evaluate the model's performance
cm = confusion_matrix(y_test, model3_preds,  normalize = "true")
sns.heatmap(cm, annot=True, cmap="Greens", fmt='g')
plt.show()
np.mean(model3_preds)

print(classification_report(y_test, model3_preds))

# AWFUL.

################################################
# Using Pipeline: Improving Model 1
# add back the topics, play with parameters,
# increase the test size
################################################

X = full_df[['comments', 'sentiment', 'dale_chall', 'type_token_ratio',
       'characters', 'syllables', 'words', 'wordtypes', 'sentences',
       'paragraphs', 'long_words', 'complex_words', 'complex_words_dc',
       'tobeverb', 'auxverb', 'conjunction', 'pronoun', 'preposition',
       'nominalization', 'topic_0', 'topic_1', 'topic_2',
        'topic_3', 'topic_4']] # add back the topics.
# NOTE: there are some NAN in X. We will do nearest neighbor interpolation
# documentation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate
X = X.interpolate(method="nearest")

high = merged_df['high']
# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    high,
                                                    test_size=0.25,
                                                    random_state=410)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('PCA', PCA()),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

parameters = {'PCA__n_components': [1, 3, 5, 7],
              'classifier__C': [0.0001, 0.01, 0.1],# regularization
              'classifier__class_weight': ['balanced', {0: 1.33, 1:4.027}]}
# len(y_train)/y_train.value_counts()

grid_search = GridSearchCV(pipeline,
                           parameters,
                           n_jobs = -1,
                           cv = 10,
                           verbose = 1)
# Fitting 5 folds for each of 64 candidates, totalling 320 fits

grid_search.fit(X_train,
                y_train)

grid_search.best_score_ # 0.7499

grid_search.best_estimator_

pd.DataFrame(grid_search.cv_results_)

# assessing fit of best estimator
model4 = grid_search.best_estimator_

model4_preds = model4.predict(X_test)

acc_train = np.mean(model4.predict(X_train) == y_train)
acc = np.mean(model4_preds == y_test) # Out[151]: for 5 components 0.7533 nice..
print('Accuracy: ' + str(acc))

#pd.DataFrame(np.transpose(model4['classifier'].coef_),
#             index = X_train.keys())

# Evaluate the model's performance
cm = confusion_matrix(y_test, model4_preds,  normalize = "true")
sns.heatmap(cm, annot=True, cmap="Greens", fmt='g')
plt.show()
np.mean(model4_preds)

print(classification_report(y_test, model4_preds))
