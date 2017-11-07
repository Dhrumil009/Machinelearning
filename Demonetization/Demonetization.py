import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from IPython.display import display
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import sys
import string
from twython import Twython
pd.options.mode.chained_assignment = None


def wordcloud_by_province_Demonetization(tweets):
    stopwords = set(STOPWORDS)
    stopwords.add("https")
    stopwords.add("00A0")
    stopwords.add("00BD")
    stopwords.add("00B8")
    stopwords.add("ed")
    stopwords.add("demonetization")
    stopwords.add("Demonetization co")
    stopwords.add("lakh")
    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets['text_new'].str.upper()]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Demonetization")
    plt._show()

def wordcloud_by_province_terrorists(tweets):
    a = pd.DataFrame(tweets['text'].str.contains("terrorists").astype(int))
    b = list(a[a['text']==1].index.values)
    stopwords = set(STOPWORDS)
    stopwords.add("https")
    stopwords.add("terrorists")
    stopwords.add("00A0")
    stopwords.add("00BD")
    stopwords.add("00B8")
    stopwords.add("ed")
    stopwords.add("demonetization")
    stopwords.add("Demonetization co")
    stopwords.add("lakh")
    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets.ix[b,:]['text_new'].str.upper()]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Tweets with word 'terrorists'")
    plt._show()


def wordcloud_by_province_narendramodi(tweets):
    a = pd.DataFrame(tweets['text'].str.contains("narendramodi").astype(int))
    b = list(a[a['text']==1].index.values)
    stopwords = set(STOPWORDS)
    stopwords.add("narendramodi")
    stopwords.add("https")
    stopwords.add("00A0")
    stopwords.add("00BD")
    stopwords.add("00B8")
    stopwords.add("ed")
    stopwords.add("demonetization")
    stopwords.add("Demonetization co")
    stopwords.add("lakh")
    wordcloud = WordCloud(background_color="white",stopwords=stopwords,random_state = 2016).generate(" ".join([i for i in tweets.ix[b,:]['text_new'].str.upper()]))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("Tweets with word 'narendramodi'")
    plt._show()


def Tweets_by_Time(tweets):
    print(tweets['retweetCount'].describe())

    tweets['hour'] = pd.DatetimeIndex(tweets['created']).hour
    tweets['date'] = pd.DatetimeIndex(tweets['created']).date
    tweets['minute'] = pd.DatetimeIndex(tweets['created']).minute

    tweets_hour = tweets.groupby(['hour'])['retweetCount'].sum()
    tweets_minute = tweets.groupby(['minute'])['retweetCount'].sum()
    tweets['text_len'] = tweets['text'].str.len()
    tweets_avgtxt_hour = tweets.groupby(['hour'])['text_len'].mean()

    # tweets_hour.transpose().plot(kind='line',figsize=(6.5, 4))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.title('The number of retweet by hour')
    #
    # tweets_minute.transpose().plot(kind='line',figsize=(6.5, 4))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.title('The number of retweet by minute')

    tweets_avgtxt_hour.transpose().plot(kind='line',figsize=(6.5, 4))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('The Average of lenght by hour')
    plt.show()

def Number_by_Source(tweets):

    tweets['statusSource_new'] = ''

    for i in range(len(tweets['statusSource'])):
        m = re.search('(?<=>)(.*)', tweets['statusSource'][i])
        try:
            tweets['statusSource_new'][i] = m.group(0)
        except AttributeError:
            tweets['statusSource_new'][i] = tweets['statusSource'][i]

    # print(tweets['statusSource_new'].head())

    tweets['statusSource_new'] = tweets['statusSource_new'].str.replace('</a>', ' ', case=False)



    tweets['statusSource_new'] = tweets['statusSource_new'].str.replace('</a>', ' ', case=False)
    #print(tweets[['statusSource_new','retweetCount']])

    tweets_by_type= tweets.groupby(['statusSource_new'])['retweetCount'].sum()
    #print(tweets_by_type)

    tweets_by_type.transpose().plot(kind='bar',figsize=(10, 5))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Number of retweetcount by Source')
    plt._show()


def Number_by_Source_bis(tweets):
    tweets['statusSource_new2'] = ''

    for i in range(len(tweets['statusSource_new'])):
        if tweets['statusSource_new'][i] not in ['Twitter for Android ','Twitter Web Client ','Twitter for iPhone ']:
            tweets['statusSource_new2'][i] = 'Others'
        else:
            tweets['statusSource_new2'][i] = tweets['statusSource_new'][i]
    #print(tweets['statusSource_new2'])

    tweets_by_type2 = tweets.groupby(['statusSource_new2'])['retweetCount'].sum()



    tweets_by_type2.transpose().plot(kind='pie',figsize=(6.5, 4))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Number of retweetcount by Source bis')
    plt._show()

def word_feats(tweets):
    negids = ['negative', 'poor move', 'cheating', 'fraud', 'bribe', 'black money', 've mudrikaran', 'kala dhan', 'deaths',
              'crisis', 'queues']
    posids = ['anti curroption', 'white money', 'justice', 'good move', 'positive', 'development', 'anti terrorism']

    negids = np.array(negids)
    posids = np.array(posids)
    negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = len(negfeats) * 3 / 4
    poscutoff = len(posfeats) * 3 / 4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print
    'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    classifier = NaiveBayesClassifier.train(trainfeats)
    print
    'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    classifier.show_most_informative_features()

def main():
    tweets = pd.read_csv('demonetization-tweets.csv', encoding="ISO-8859-1")
    display(tweets.head(3))

    tweets['text_new'] = ''

    for i in range(len(tweets['text'])):
        m = re.search('(?<=:)(.*)', tweets['text'][i])
        try:
            tweets['text_new'][i] = m.group(0)
        except AttributeError:
            tweets['text_new'][i] = tweets['text'][i]

    print(tweets['text_new'].head())

    wordcloud_by_province_Demonetization(tweets)
    wordcloud_by_province_terrorists(tweets)
    wordcloud_by_province_narendramodi(tweets)
    Tweets_by_Time(tweets)
    Number_by_Source(tweets)
    Number_by_Source_bis(tweets)
    word_feats(tweets)

if __name__ == '__main__':
    main()
