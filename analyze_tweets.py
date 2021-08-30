import networkx as nx
from collections import Counter, OrderedDict
from twitter_preprocessor import TwitterPreprocessor
from pymongo import MongoClient
from tqdm import tqdm
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk import pos_tag, word_tokenize
import os
from dotenv import load_dotenv
import datetime

load_dotenv()

client = MongoClient(host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT")), username=os.getenv("DB_USERNAME"), password=os.getenv("DB_PASSWORD"), authSource=os.getenv("DB_AUTH_DB"))
db = client[os.getenv("DB_AUTH_DB")]
col = db[os.getenv("DB_COLLECTION")]


class TweetAnalysis:

    def __init__(self, collection):
        self.collection = collection


    def tweets(self):
        """
            Tweets function reads all the records from a MongoDB
            :return: a list of all tweet objects in our DB
        """
        tweets = []
        iterator = self.collection.find()
        print('...Collecting Tweet Objects...')
        for i in tqdm(iterator):
            tweets.append(i)
        return tweets

    def hashtags(self):
        """
        hashtags function collects all the hashtags included in the list of tweets
        :return: a list of all hashtags included in tweets
        """
        hashtagList=[]
        for t in self.tweets():
            tags = t['entities']['hashtags']
            for j in tags:
                hashtagList.append(j['text'])
            if 'extended_tweet' in t:
                extended_tags = t['extended_tweet']['entities']['hashtags']
                for n in extended_tags:
                    hashtagList.append(n['text'])
        return hashtagList

    def urls(self):
        """
        urls function collects all the urls included in the list of tweets
        :return: a list of all urls included in tweets
        """
        urlList = []
        for t in self.tweets():
            urls = t['entities']['urls']
            for j in urls:
                urlList.append(j['expanded_url'])
            if 'extended_tweet' in t:
                extended_urls = t['extended_tweet']['entities']['urls']
                for n in extended_urls:
                    urlList.append(n['expanded_url'])
        return urlList

    def mentions(self):
        """
        mentions function collects all the mentions included in the list of tweets
        :return: a list of all mentions included in tweets
        """
        userMentions = []
        for t in self.tweets():
            urls = t['entities']['user_mentions']
            for j in urls:
                userMentions.append(j['id_str'])
            if 'extended_tweet' in t:
                extended_mentions = t['extended_tweet']['entities']['user_mentions']
                for n in extended_mentions:
                    userMentions.append(n['id_str'])
        return userMentions

    def users(self):
        """
        users function collects all the users included in the list of tweets
        :return: a list of all users ids that posted the tweets
        """
        userList=[]
        for t in self.tweets():
            userList.append(t['user']['id_str'])
        return userList

    def texts(self):
        all_texts = []
        for t in self.tweets():
            if 'extended_tweet' in t:
                all_texts.append(t['extended_tweet']['full_text'])
            else:
                all_texts.append(t['text'])
        return all_texts

    def retweets(self):
        rts = []
        for t in self.tweets():
            if 'retweeted_status' in t:
                rts.append(t)
        return rts

    def only_tweets(self):
        ts = []
        for t in self.tweets():
            if 'retweeted_status' not in t:
                ts.append(t)
        return ts

    def text_only_tweets(self):
        all_texts = []
        for t in self.only_tweets():
            if 'extended_tweet' in t:
                all_texts.append(t['extended_tweet']['full_text'])
            else:
                all_texts.append(t['text'])
        return all_texts

    def text_retweets(self):
        all_texts = []
        for t in self.retweets():
            if 'extended_tweet' in t:
                all_texts.append(t['extended_tweet']['full_text'])
            else:
                all_texts.append(t['text'])
        return all_texts

    def hashtagCloud(self):
        print ('...Generating Tag Cloud...')
        tags = self.hashtags()
        text = (" ").join(tags)
        wordcloud = WordCloud(width = 1500, height = 1500,random_state=1, background_color='black', colormap='Set2',
                              collocations=False, stopwords=STOPWORDS).generate(text)
        # Set figure size
        # Display image
        plt.imshow(wordcloud,interpolation='bilinear')
        # No axis details
        plt.axis("off")
        plt.savefig('figures/hashtagCloud_'+str(len(tags))+'.png', dpi=300)

    def textCloud(self):
        texts = self.texts()
        wordList=[]
        for t in texts:
            p = TwitterPreprocessor(t)
            p.fully_preprocess()
            new = p.text
            text = word_tokenize(new)
            wordList.extend(text)
        wordCloudText = (" ").join(wordList)
        wordcloud = WordCloud(width=1500, height=1500, random_state=1, background_color='black', colormap='Set2',
                              collocations=False, stopwords=STOPWORDS).generate(wordCloudText)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('figures/WordCloud_' + str(len(wordList)) + '.png', dpi=300)

    def hashtagFrequenciesCsv(self):
        tags = self.hashtags()
        c = Counter(tags)
        y = OrderedDict(c.most_common())
        file = open('files/hashtagFrequencies_'+datetime.date.today().strftime("%B %d, %Y")+'.csv','w',encoding='utf-8')
        for k,v in y.items():
            file.write(str(k)+','+str(v)+'\n')
        file.close()
        return c

    def wordFrequenciesCsv(self):
        texts = self.texts()
        wordList = []
        for t in texts:
            p = TwitterPreprocessor(t)
            p.fully_preprocess()
            new = p.text
            text = word_tokenize(new)
            wordList.extend(text)
        c = Counter(wordList)
        y = OrderedDict(c.most_common())
        file = open('files/WordFrequencies_'+datetime.date.today().strftime("%B %d, %Y")+'.csv','w',encoding='utf-8')
        for k,v in y.items():
            file.write(str(k)+','+str(v)+'\n')
        file.close()
        return c

    def most_retweeted_tweets(self,n):
        rts = self.retweets()
        rtdict={}
        for r in rts:
            rtcount = r['retweeted_status']['retweet_count']
            rtdict[r['id_str']]=rtcount
        top = sorted(rtdict, key=rtdict.get, reverse=True)[:n]
        topUrls = ["https://twitter.com/user/status/"+i for i in top]
        file = open('files/top_'+str(n)+'_Retweets_' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',
                    encoding='utf-8')
        for k in topUrls:
            file.write(str(k)+ '\n')
        file.close()
        return topUrls

    def most_favorited_tweets(self,n):
        ts = self.tweets()
        tdict={}
        for t in ts:
            if 'retweeted_status' in t:
                favcount = t['retweeted_status']['favorite_count']
                tdict[t['id_str']]=favcount
            else:
                favcount = t['favorite_count']
                tdict[t['id_str']] = favcount
        top = sorted(tdict, key=tdict.get, reverse=True)[:n]
        topFavs = ["https://twitter.com/user/status/"+i for i in top]
        file = open('files/top_'+str(n)+'_Favorite_tweets_' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',
                    encoding='utf-8')
        for k in topFavs:
            file.write(str(k)+ '\n')
        file.close()
        return topFavs

t = TweetAnalysis(col)
