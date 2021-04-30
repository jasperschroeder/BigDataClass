# Scraping.py

###################
# CHANGED VERSION #
###################



# We go through the push API to get the ids of every post, then use the official reddit
# API to get the contents of each post and metadata of interest.
import requests
import json
import pandas as pd
import time
import math

#first_epoch = 1370000000 # Right before the first post in 2012
first_epoch = 1451606400 # January 1, 2016
last_epoch = 1467331200 # July 1, 2016

def getPushshiftData(after, before):
    url = 'https://api.pushshift.io/reddit/submission/search/'
    sort = '?sort_type=created_utc&sort=asc'
    subr = '&subreddit=Bitcoin'
    after = '&after=' + str(after)
    before = '&before=' + str(before)
    size = '&size=100'
    full_url = url + sort + subr + after + before + size
    print(full_url)
    r = requests.get(full_url)
    data = json.loads(r.text)
    return data['data']


def scraping(first_epoch, last_epoch, filename):
    timestamps = []
    authors = []
    scores = []
    comments = []
    ids = []
    titles = []
    texts = []

    after = first_epoch
    while int(after) < last_epoch:
        data = getPushshiftData(after,last_epoch)
        tmp_times = []; tmp_authors = []; tmp_scores = [];
        tmp_coms = []; tmp_ids = []; tmp_titles = [];
        tmp_texts = [];

        for post in data:
            tmp_times.append(post['created_utc'])
            tmp_authors.append(post['author'])
            tmp_scores.append(post['score'])
            tmp_coms.append(post['num_comments'])
            tmp_ids.append(post['id'])
            tmp_titles.append(post['title'])
            try:
                tmp_texts.append(post['selftext'])
            except:
                tmp_exts.append(math.nan)

        try:
            if max(tmp_times) not in timestamps:
                timestamps = timestamps + tmp_times
                authors = authors + tmp_authors
                scores = scores + tmp_scores
                comments = comments + tmp_coms
                ids = ids + tmp_ids
                titles = titles + tmp_titles
                texts = texts + tmp_texts
            else:
                break
        except:
            break


        after = max(timestamps)
        # if after == previous_last:
        #     break
        # previous_last = max(timestamp)



        print([str(len(ids)) + " posts collected so far."])
        time.sleep(5)

    # Write to a csv file
    d = {'id':ids, 'timestamp':timestamps, 'author':authors,
        'score':scores, 'comments':comments,# 'sticks':sticks,
        'title':titles, 'text':texts}
    df = pd.DataFrame(d)
    df.to_csv(filename, index=False)

#scraping(1451606400, 1467331200, filename="redditjanjun2016.csv")
#scraping(1467331200, 1483228800, filename="redditjuldec2016.csv")


#scraping(1483228800, 1498867200, filename="redditjanjun2017.csv")
scraping(1498867200, 1514764800, filename="redditjuldec2017.csv")


#scraping(1514764800, 1530403200, filename="redditjanjun2018.csv")
#scraping(1530403200, 1546300800, filename="redditjuldec2018.csv")


#scraping(1546300800, 1561939200, filename="redditjanjun2019.csv")
#scraping(1561939200, 1577836800, filename="redditjuldec2019.csv")


#scraping(1577836800, 1593561600, filename="redditjanjun2020.csv")
#scraping(1593561600, 1609459200, filename="redditjuldec2020.csv")





#
