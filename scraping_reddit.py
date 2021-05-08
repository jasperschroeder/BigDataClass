# Scraping.py

###################
# CHANGED VERSION #
###################

import requests
import json
import pandas as pd
import time
import math

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
                tmp_texts.append(math.nan)

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

        print([str(len(ids)) + " posts collected so far."])
        time.sleep(3)

    # Write to a csv file
    d = {'id':ids, 'timestamp':timestamps, 'author':authors,
        'score':scores, 'comments':comments, # 'sticks':sticks,
        'title':titles, 'text':texts}
    df = pd.DataFrame(d)
    df.to_csv(filename, index=False)

# scraping(1451606400, 1467331200, filename="redditjanjun2016.csv")     # 1
# scraping(1467331200, 1483228800, filename="redditjuldec2016.csv")     # 2

# scraping(1483228800, 1498867200, filename="redditjanjun2017.csv")     # 3
# scraping(1498867200, 1514764800, filename="redditjuldec2017.csv")     # 4

# scraping(1514764800, 1528963814, filename="redditjanjun2018a.csv")     # 5a
# scraping(1528963814, 1530403200, filename="redditjanjun2018b.csv")     # 5b
# scraping(1530403200, 1546300800, filename="redditjuldec2018.csv")     # 6

# scraping(1546300800, 1561939200, filename="redditjanjun2019.csv")     # 7
# scraping(1561939200, 1577836800, filename="redditjuldec2019.csv")     # 8

# scraping(1577836800, 1593561600, filename="redditjanjun2020.csv")     # 9
# scraping(1593561600, 1609459200, filename="redditjuldec2020.csv")     # 10

#scraping(1609459200, 1619827200, filename="redditjanapr2021.csv")     # 11

datasets = ['redditjanjun2016', 'redditjuldec2016',
    'redditjanjun2017', 'redditjuldec2017',
    'redditjanjun2018a', 'redditjanjun2018b', 'redditjuldec2018',
    'redditjanjun2019', 'redditjuldec2019',
    'redditjanjun2020', 'redditjuldec2020']
    #, 'redditjanapr2021']

final_df = pd.DataFrame()

for elem in datasets:
    csv = elem + '.csv'
    df = pd.read_csv(csv)
    final_df = pd.concat([final_df, df])

final_df.to_csv('final_dataset.csv')
