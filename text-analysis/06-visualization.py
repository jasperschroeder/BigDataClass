###############################################################################
# title:        06-visualization.py
# created on:   May 27, 2021
# summary:      visualize thread information. Only Dale Chall for now
#               since we have other visualizations
###############################################################################
import pandas as pd
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('df_final.csv')
# again we remove the null values of df
missing = (df["text"].isnull()) | ((df["text"] == '[deleted]')) | ((df["text"] == '[removed]'))
df = df.loc[~missing]

# how does the Dale-Chall score look?
feature_df = pd.read_csv('nlp-data/feature_df.csv', index_col = 0)

feature_df.keys()

feature_df['dale_chall'].describe()
# example of min:
df[feature_df['dale_chall']==0]['text'].iloc[0]
df[feature_df['dale_chall']==0]['text'].iloc[10]
df[feature_df['dale_chall']==0]['text'].iloc[1]

# This is HTML / CSS!
df[feature_df['dale_chall']==np.max(feature_df['dale_chall'])]['text'].iloc[0]

# Above Median Dale Chall
df[feature_df['dale_chall']>=np.mean(feature_df['dale_chall'])]['text'].iloc[0]
