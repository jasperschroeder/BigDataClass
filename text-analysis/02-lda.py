###############################################################################
# title:        02-lda.py
# created on:   May 15, 2021
# summary:      lda, outputs topic weights in a df
###############################################################################
import pandas as pd
import numpy as np
import string
import spacy.cli
import matplotlib.pyplot as plt
import pdtext
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

df = pd.read_csv('nlp-data/clean_text_df.csv', index_col = 0)

# after cleaning some text is nan - was originally emojis perhaps
missing = df["text"].isnull()
#df = df.loc[~missing]

# stop words: english + bitcoin related + things we have found after cleaning
my_stop_words = text.ENGLISH_STOP_WORDS.union(["https", "www", "com", "bitcoin", "btc", "bitcoins",
                                              "just","like", "wallet", "btc", "blockchain",
                                               "crypto", "coinbase", "amp",
                                               "im", "iv", "id", "ive", "ampxb"])


# get the most common words
vectorizer = CountVectorizer(lowercase=True,
                             max_df = 0.3,
                             stop_words=my_stop_words,
                             min_df=0.025)

# fit the CountVectorizer to the clean text
vectorizer.fit(df2['text'])

# get the vocabulary of the text
vocab = vectorizer.get_feature_names()
len(vocab) # 241
# save this for 02-lda-topics.py
dump(vocab,'vocab.joblib')

# term frequency
tf = vectorizer.transform(df['text'])

# Compute LDA Models
for cp in [3, 5, 8, 10, 15, 20]:
    # LDA topic model
    print('Fitting LDA Model for ncomp' + str(cp) + '!!!')
    lda_model = LatentDirichletAllocation(n_components=cp,
                                          max_iter=10,
                                          evaluate_every=1,
                                          random_state=420,
                                          verbose=1)
    # fit
    lda_model.fit_transform(tf)

    # Log Likelihood: Higher the better
    #print("Log Likelihood: ", lda_model.score(tf))
    # Log Likelihood:  -26461197.1741212

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    #print("Perplexity: ", lda_model.perplexity(tf))
    # Perplexity:  396.69211197749775

    fname = 'lda-models/2021-13-mai-lda-model-stemmed-ncomp-' + str(cp) + '.joblib'
    dump(lda_model,fname)

###############################
# Plot Perplexity of LDA Models
###############################
# we know that perplexity decreases with the number of topics
# thus we use a rule-of-thumb for unsupervised learning: "elbow rule"

lst_perplexity = []
lst_loglik = []


# warning -- slow
for cp in [3, 5, 8, 10, 15, 20]:
    fname = '2021-13-mai-lda-model-stemmed-ncomp-' + str(cp) + '.joblib'
    lda_model = load(fname)
    lst_loglik.append(lda_model.score(tf))
    lst_perplexity.append(lda_model.perplexity(tf))
    print(lst_perplexity)

# plot the number of compotents vs. perplexity
plt.plot([3, 5, 8, 10, 15, 20],
         lst_perplexity,
         'r-o')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.title('LDA Model Perplexity by Number of Topics')
#plt.show()
plt.savefig('plots/2021-13-mai-perplexity-plot.png')

# plot the number of compotents vs. perplexity, exclude 20
plt.plot([3, 5, 8, 10, 15],
         lst_perplexity[0:len(lst_perplexity)-1],
         'r-o')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity')
plt.title('LDA Model Perplexity by Number of Topics')
#plt.show()
plt.savefig('plots/2021-13-mai-perplexity-plot-3to15.png')
plt.close()

# plot the number of components vs log-likelihood
plt.plot([3, 5, 8, 10, 15, 20],
         lst_loglik,
         'b-o')
plt.xlabel('Number of Topics')
plt.ylabel('Log Likelihood')
plt.title('LDA Model Log Likelihood by Number of Topics')
#plt.show()
plt.savefig('plots/2021-13-mai-loglik-plot.png')
plt.close()

# plot the number of components vs log-likelihood
plt.plot([3, 5, 8, 10, 15],
         lst_loglik[0:len(lst_loglik)-1],
         'b-o')
plt.xlabel('Number of Topics')
plt.ylabel('Log Likelihood')
plt.title('LDA Model Log Likelihood by Number of Topics')
#plt.show()
plt.savefig('plots/2021-13-mai-loglik-plot-3to15.png')
plt.close()

########################
# select final LDA model
########################

# The "elbow" is at 10 components. However, we can further reduce dimensionality
# with 5 components, and we find an interpretable output.
# To make a final decision, we will assess the performance of the topic scores
# on predictive models.

################################
# output dataset with LDA scores
################################

for cp in [5, 10]:
    fname = 'lda-models/2021-13-mai-lda-model-stemmed-ncomp-' + str(cp) + '.joblib'
    lda_model = load(fname)

    thread_topics = lda_model.transform(tf)
    # thread_topics.shape # (177277, 10) good

    topic_df = pd.DataFrame(thread_topics)

    new_cols = pd.Series(range(cp)).apply(lambda x: 'topic_'+str(x)).tolist()
    topic_df.columns = new_cols
    # rename the index to be like the original df
    topic_df.index = df.index

    # concatenate
    #df_with_topics = pd.concat([df, topic_df], axis = 1)
    #print(df_with_topics.shape) # (177277, 24)

    # write as csv
    op_fname = 'nlp-data/2021-15-mai-df_topics_' + str(cp) + '.csv'
    print(op_fname)
    topic_df.to_csv(op_fname, index=True)