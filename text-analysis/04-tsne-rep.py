###############################################################################
# title:        03-tsne-rep.py
# created on:   May 13, 2021
# summary:      t-SNE representation of documents
#               uses document features (word count, punctuation counts, etc)
#               and topic scores.
#               colored by dominant topic score
#
#               good resource for t-SNE do's and don'ts:
#               http://deeplearning.csail.mit.edu/slide_cvpr2018/laurens_cvpr18tutorial.pdf
#
# note:         very slow. I suggest you take a sample from the data
#               and run that first to get an idea
#
#               to export the plot:
#               install selenium (pip install -U selenium) and geckodriver
#               pip install geckodriver-autoinstaller
#               in python console:
#               >>> from selenium import webdriver
#               >>> import geckodriver_autoinstaller
#               >>> geckodriver_autoinstaller.install()
#
#               source: https://pypi.org/project/geckodriver-autoinstaller/
###############################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
import matplotlib.colors as mcolors
from bokeh.models import Label
from selenium import webdriver
import geckodriver_autoinstaller
geckodriver_autoinstaller.install()
from bokeh.io import export_png

def tSNE_plot(n_topic):
       input_fname = 'nlp-data/df_topics_' + str(n_topic) + '.csv'
       tsne_df = pd.read_csv(input_fname, index_col = 0)

       keep = pd.Series(range(n_topic)).apply(lambda x: 'topic_'+str(x)).tolist()

       tsne_df = tsne_df[keep]
       # scale: best to do this before t-SNE
       #scaler = StandardScaler()
       #scaled_tsne_df = scaler.fit_transform(tsne_df)

       # scaled_tsne_df.shape # 177277, 26 ok good

       # get the dominant topic of each document
       dominant_topic = tsne_df[['topic_' + str(c) for c in range(n_topic)]].idxmax(axis=1)

       # extract the number from dominant topic
       topic_num = dominant_topic.apply(lambda x: int(x.split('_')[1]))
       topic_num = topic_num.to_numpy()

       # tSNE Dimension Reduction -- warning: slow
       # source: https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
       tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
       tsne_lda = tsne_model.fit_transform(tsne_df)

       # plot, color by topic number (for now)
       n_topics = n_topic
       output_fname = 'plots/2021-15-mai-tsne-plot-' + str(n_topic) + '-topics.png'
       mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
       plot = figure(title="t-SNE Clustering, colored by {} LDA Topics".format(n_topics),
              plot_width=900, plot_height=700)
       plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
       export_png(plot, filename=output_fname)
       return

# make the plots
tSNE_plot(n_topic=5)
tSNE_plot(n_topic=10)