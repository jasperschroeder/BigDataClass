import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
import matplotlib.colors as mcolors
from bokeh.models import Label
from selenium import webdriver
import geckodriver_autoinstaller
geckodriver_autoinstaller.install()
from bokeh.io import export_png

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
keep = ['author', 'comments', 'sentiment', 'dale_chall',
       'type_token_ratio', 'characters', 'syllables', 'words', 'wordtypes',
       'sentences', 'paragraphs', 'long_words', 'complex_words',
       'complex_words_dc', 'tobeverb', 'auxverb', 'conjunction', 'pronoun',
       'preposition', 'nominalization', 'topic_0', 'topic_1', 'topic_2',
        'topic_3', 'topic_4']

full_df = full_df[keep]

# get a list of reccuring users for clustering.
# we define recurring to mean that they post at least 5 times
value_counts = df['author'].value_counts()
df_users = pd.DataFrame(value_counts.to_numpy(), index=value_counts.index, columns=['n_thread'])
recurring_users = df_users[df_users['n_thread']>=3].index

# we have 5835 recurring users

# keep only the threads from recurring users
full_df = full_df[full_df['author'].apply(lambda x: x in recurring_users)]
full_df_by_author = full_df.groupby('author').agg('mean')

# get the most common topic by author
df['dominant_topic'] = dominant_topic
topic_by_author = df[['author', 'dominant_topic']]
dominant_topic_by_author = topic_by_author.groupby('author').agg(func=lambda x:x.value_counts().idxmax())

# keep only the recurring authors
dominant_topic_by_author['author'] = dominant_topic_by_author.index
dominant_topic_by_author = dominant_topic_by_author[dominant_topic_by_author['author'].apply(lambda x: x in recurring_users)]
dominant_topic_by_author = dominant_topic_by_author['dominant_topic']

# convert to numeric
topic_num = dominant_topic_by_author.apply(lambda x: int(x.split('_')[1]))
topic_num = topic_num.to_numpy()

# sentiment: positive, neutral, negative
def sentiment_group(sent):
       if sent >= 0.05:
              return(3) # positive
       elif sent <= -0.05:
              return(1) # negative
       else:
              return(2) # neutral

author_sentiment = full_df_by_author['sentiment'].apply(sentiment_group)


# we will visualize the data from full_df_by_author with t-SNE, colored by
# dominant_topic_by_author

# t-SNE embeddings
# scale: best to do this before t-SNE
scaler = StandardScaler()
scaled_df = scaler.fit_transform(full_df_by_author)

scaled_df.shape # (5835, 24)

# tSNE Dimension Reduction -- warning: slow
# source: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.95, init='pca',
                  perplexity=50)
tsne_lda = tsne_model.fit_transform(scaled_df)

# plot, color by topic number (for now)
#n_topics = 5

n_topics = 10
output_fname = 'plots/author-tsne-plot-sentiment.png'
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE of Recurring Authors, colored by Sentiment",
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[author_kmeans])
export_png(plot, filename=output_fname)

#####
# PCA: not good - 2 components cannot represent the data well
#####
#pca = PCA(2)  # project from 64 to 2 dimensions
#projected = pca.fit_transform(scaled_df)
#print(scaled_df.shape)
#print(projected.shape)

#plt.scatter(projected[:, 0], projected[:, 1],
#            c=author_sentiment, edgecolor='none', alpha=0.5,
#            cmap=plt.cm.get_cmap('viridis', 10))
#plt.xlabel('component 1')
#plt.ylabel('component 2')
#plt.colorbar();
#plt.show()

##################################
# author subset selection
##################################
# start with the 100 most recurring users

# sentiment df - rows are author, columns are date/time and sentiment of the post
sentiment_df = df[df['author'].apply(lambda x: x in recurring_users[range(5)])][['author', 'Day', 'timestamp','sentiment']]

# combine the date and timestamp
sentiment_df['time'] = sentiment_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d-%H:%M:%S'))
del(sentiment_df['timestamp'])

# line plot
import seaborn as sns
sns.lineplot(data=sentiment_df, x="time", y="sentiment", hue="author")
plt.show()

author_df = sentiment_df.pivot_table(index='author', columns='Day', values='sentiment')


# can we predict bitcoin volatility with author sentiment?
# aggregate sentiment by day


############################
# Arrange author df by time
############################
# issue - far too many NA -- we cannot analyze anything

# user value counts
#value_counts = df['author'].value_counts()
#df_users = pd.DataFrame(value_counts.to_numpy(), index=value_counts.index, columns=['n_thread'])
#recurring_users = df_users[df_users['n_thread']>=10].index

# sentiment df - rows are author, columns are date/time and sentiment of the post
#sentiment_df = df[df['author'].apply(lambda x: x in recurring_users)][['author', 'Day', 'timestamp','sentiment']]

# combine the date and timestamp
#sentiment_df['time'] = sentiment_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d-%H:%M:%S'))
#del(sentiment_df['Day'])
#del(sentiment_df['timestamp'])

#author_df = sentiment_df.pivot_table(index='author', columns='time', values='sentiment')

# export to csv for KmL clustering
#author_df.to_csv('nlp-data/author_df.csv', index=True)