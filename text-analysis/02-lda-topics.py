###############################################################################
# title:        01-lda-topics.py
# created on:   May 13, 2021
# summary:      visualizes topics of LDA model
#               NOTE: this can only be run after 01-thread-info.py
###############################################################################

# import libraries
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from joblib import dump, load


# extract topic information
# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

def plot_top_words(model, feature_names, n_top_words, title, fig_fname):
    fig, axes = plt.subplots(1, 5, figsize=(30, 12), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.5)
    #plt.savefig('plots/2021-13-mai-topics-in-lda-5-cp.png')
    plt.savefig(fig_fname)
    #plt.show()

# load the vocabulary we got from 01-thread-info.py
vocab = load('vocab.joblib')

# plot for LDA model with 5 and 10 topics
for cp in [5,10]:
    # load the LDA model
    fname = 'lda-models/2021-13-mai-lda-model-stemmed-ncomp-' + str(cp) + '.joblib'
    lda_model = load(fname)

    plot_top_words(lda_model,
                   feature_names=vocab,
                   n_top_words=10,
                   title = 'Topics in LDA model',
                   fig_fname= 'plots/2021-13-mai-topics-in-lda-' + str(cp) +'-cp.png')
