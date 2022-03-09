import networkx as nx
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
from collections import Counter, OrderedDict
from twitter_preprocessor import TwitterPreprocessor
from pymongo import MongoClient
from tqdm import tqdm
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from wordcloud import ImageColorGenerator
from nltk import pos_tag, word_tokenize
import os
from dotenv import load_dotenv
import datetime
import pickle
from PIL import Image
import numpy as np
from pprint import pprint
load_dotenv()
stemmer = SnowballStemmer('english')
# nltk.download('wordnet')

client = MongoClient(host=os.getenv("DB_HOST"), port=int(os.getenv("DB_PORT")), username=os.getenv("DB_USERNAME"), password=os.getenv("DB_PASSWORD"), authSource=os.getenv("DB_AUTH_DB"))
db = client[os.getenv("DB_AUTH_DB")]
col = db[os.getenv("DB_COLLECTION")]
resultsCol=db['ukraine_results']

def lemmatize_stemming(tweetText):
    return stemmer.stem(WordNetLemmatizer().lemmatize(tweetText, pos='v'))


def preprocess(tweetText):
    result = []
    for token in gensim.utils.simple_preprocess(tweetText):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def get_tweets():
    """
    Tweets function reads all the records from a MongoDB
    :return: a list of all tweet objects in our DB
    """
    tweets = []
    iterator = col.find().limit(100000)
    print('...Collecting Tweet Objects...')
    for i in tqdm(iterator):
        tweets.append(i)
    return tweets

class TweetAnalysis:

    def __init__(self, collection,tweets):
        self.collection = collection
        self.tweets = tweets

    # def tweets(self):
    #     """
    #         Tweets function reads all the records from a MongoDB
    #         :return: a list of all tweet objects in our DB
    #     """
    #     tweets = []
    #     iterator = self.collection.find()
    #     print('...Collecting Tweet Objects...')
    #     for i in tqdm(iterator):
    #         tweets.append(i)
    #     return tweets

    def hashtags(self):
        """
        hashtags function collects all the hashtags included in the list of tweets
        :return: a list of all hashtags included in tweets
        """
        hashtagList=[]
        for t in self.tweets:
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
        for t in self.tweets:
            urls = t['entities']['urls']
            for j in urls:
                urlList.append(j['expanded_url'])
            if 'extended_tweet' in t:
                extended_urls = t['extended_tweet']['entities']['urls']
                for n in extended_urls:
                    urlList.append(n['expanded_url'])
        return urlList

    def top_n_urls(self,n):
        '''
        function used to find the n top urls - number of count
        :return: a list of the expanded urls in decreasing order - doesn't return count
        '''
        urlList =self.urls()
        most_common = [u for u, u_count in Counter(urlList).most_common(n)]
        most_common_dict = Counter(urlList).most_common(n)
        file = open('files/UKRAINE_most_active_urls_' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',
                    encoding='utf-8')
        for k in most_common:
            file.write(str(k) + '\n')
        file.close()
        return most_common,most_common_dict

    def mentions(self):
        """
        mentions function collects all the mentions included in the list of tweets
        :return: a list of all mentions included in tweets
        """
        userMentions = []
        for t in self.tweets:
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
        for t in self.tweets:
            userList.append(t['user']['id_str'])
        return userList

    def most_active_users(self,n):
        """
        most active users function collects the n most active users included in the list of tweets
        :return: a list of the n most active users
        """
        userList = []
        for t in self.tweets:
            userList.append(t['user']['screen_name'])
        most_common_dict = Counter(userList).most_common(n)
        most_common = [u for u, u_count in Counter(userList).most_common(n)]
        file = open('files/UKRAINE_most_active_users_' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',
                    encoding='utf-8')
        for k in most_common:
            file.write("https://twitter.com/" +str(k) + '\n')
        file.close()
        return most_common,most_common_dict

    def texts(self):
        """
        texts function collects all the texts in the list of tweets
        :return: a list of all texts
        """
        all_texts = []
        for t in self.tweets:
            if 'extended_tweet' in t:
                all_texts.append(t['extended_tweet']['full_text'])
            else:
                all_texts.append(t['text'])
        return all_texts

    def retweets(self):
        """
        retweets function collects all the tweets that are retweets
        :return: a list of all retweets
        """
        rts = []
        for t in self.tweets:
            if 'retweeted_status' in t:
                rts.append(t)
        return rts

    def only_tweets(self):
        """
        only tweets function collects all the tweets that are retweets
        :return: a list of only tweets (retweets are filtered out)
        """
        ts = []
        for t in self.tweets:
            if 'retweeted_status' not in t:
                ts.append(t)
        return ts

    def text_only_tweets(self):
        """
        text only tweets function collects the texts of only tweets
        :return: a list of all tweets' text, no retweets included
        """
        all_texts = []
        for t in self.only_tweets():
            if 'extended_tweet' in t:
                all_texts.append(t['extended_tweet']['full_text'])
            else:
                all_texts.append(t['text'])
        return all_texts

    def text_retweets(self):
        """
        text retweets function collects the texts of only retweets
        :return: a list of all retweets' text, no tweets included
        """
        all_texts = []
        for t in self.retweets():
            if 'extended_tweet' in t:
                all_texts.append(t['extended_tweet']['full_text'])
            else:
                all_texts.append(t['text'])
        return all_texts

    def hashtagCloud(self):
        """
        hashtagCloud function creates a hashtag wordcloud image
        :return: image of hashtag cloud
        """
        print ('...Generating Tag Cloud...')
        alltags = self.hashtags()
        tags = [tag.lower() for tag in alltags]
        # print (tags)
        file = open('tags.txt','w',encoding='utf-8')
        for t in tags:
            file.write(str(t)+',')
        file.close()
        text = (" ").join(tags)
        mask = np.array(Image.open('figures/viral.jpg'))
        wordcloud = WordCloud(random_state=1,background_color='black', width=1000,
               height=1000,collocations=False, colormap = "Set1",stopwords=STOPWORDS).generate(text)
        # Set figure size
        # Display image
        plt.imshow(wordcloud,interpolation='bilinear')
        # No axis details
        plt.axis("off")
        plt.savefig('figures/UKRAINE_hashtagCloud_'+str(len(tags))+'.png', dpi=300)

    def textCloud(self):
        """
        textCloud function creates a text wordcloud image
        :return: image of text cloud
        """
        alltexts = self.texts()
        texts = [text.lower() for text in alltexts]
        wordList=[]
        for t in texts:
            p = TwitterPreprocessor(t)
            p.fully_preprocess()
            new = p.text
            text = word_tokenize(new)
            wordList.extend(text)
        file = open('words.txt', 'w', encoding='utf-8')
        for t in wordList:
            file.write(str(t) + ',')
        file.close()
        wordCloudText = (" ").join(wordList)
        wordcloud = WordCloud(width=1500, height=1500, random_state=1, background_color='black', colormap='Set2',
                              collocations=False, stopwords=STOPWORDS).generate(wordCloudText)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('figures/UKRAINE_WordCloud_' + str(len(wordList)) + '.png', dpi=300)

    def hashtagFrequenciesCsv(self):
        """
        hashtagfrequencies function creates a Counter dict of the list of hashtags
        :return: counter dict of hashtags
        """
        tags = self.hashtags()
        c = Counter(tags)
        y = OrderedDict(c.most_common())
        file = open('files/hashtagFrequencies_'+datetime.date.today().strftime("%B %d, %Y")+'.csv','w',encoding='utf-8')
        for k,v in y.items():
            file.write(str(k)+','+str(v)+'\n')
        file.close()
        return c

    def wordFrequenciesCsv(self):
        """
        wordfrequencies function creates a Counter dict of the list of words
        :return: counter dict of words
        """
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
        file = open('files/UKRAINE_WordFrequencies_'+datetime.date.today().strftime("%B %d, %Y")+'.csv','w',encoding='utf-8')
        for k,v in y.items():
            file.write(str(k)+','+str(v)+'\n')
        file.close()
        return c

    def most_retweeted_tweets(self,n):
        """
        most retweeted tweets function returns a list of the top retweets urls and a list of the top retweets ids
        :return: list of strings urls, list of strings retweet ids
        """
        rts = self.retweets()
        rtdict={}
        for r in rts:
            rtcount = r['retweeted_status']['retweet_count']
            print (rtcount)
            rt_id = r['retweeted_status']['id_str']
            if rt_id in rtdict:
                print ('not here')
                if rtcount > rtdict[rt_id]:
                    rtdict[rt_id]=rtcount
            else:
                rtdict[rt_id] = rtcount
        top = sorted(rtdict, key=rtdict.get, reverse=True)[:n]
        print (top)
        toprts = ["https://twitter.com/user/status/"+i for i in top]
        file = open('files/UKRAINE_top_'+str(n)+'_Retweets_' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',
                    encoding='utf-8')
        for k in toprts:
            file.write(str(k)+ '\n')
        file.close()
        return toprts,top

    def most_favorited_tweets(self,n):
        """
        most favorited tweets function returns a list of the top favorited tweets and a list of the top favorited tweet ids
        :return: list of strings urls, list of strings most favorite ids
        """
        ts = self.tweets
        tdict={}
        for t in ts:
            if 'retweeted_status' in t:
                favcount = t['retweeted_status']['favorite_count']
                rt_id = t['retweeted_status']['id_str']
                if rt_id in tdict:
                    if favcount > tdict[rt_id]:
                        tdict[rt_id]=favcount
                else:
                    tdict[rt_id] = favcount
            else:
                favcount = t['favorite_count']
                tdict[t['id_str']] = favcount
        top = sorted(tdict, key=tdict.get, reverse=True)[:n]
        topFavs = ["https://twitter.com/user/status/"+i for i in top]
        file = open('files/UKRAINE_top_'+str(n)+'_Favorite_tweets_' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',
                    encoding='utf-8')
        for k in topFavs:
            file.write(str(k)+ '\n')
        file.close()
        return topFavs,top

    def topic_models(self,topicsNum):
        print ('...Generating topic models...')
        processed_docs = []
        for t in self.texts():
            p = TwitterPreprocessor(t)
            p.fully_preprocess()
            new = p.text
            processed_text = preprocess(new)
            processed_docs.append(processed_text)
        dictionary = gensim.corpora.Dictionary(processed_docs)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        from gensim import corpora, models
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=topicsNum, id2word=dictionary, passes=2, workers=2)
        file1 = open('files/bow_topic_models' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',encoding='utf-8')
        allTopics = {}
        for idx, topic in lda_model.print_topics(-1,15):
            allTopics[str(idx)] = dict([(str(dictionary[wid]), str(s)) for (wid, s) in lda_model.get_topic_terms(idx)])
            file1.write('Topic: {} \nWords: {}'.format(idx, topic))
            file1.write('\n')
        file1.close()
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=topicsNum, id2word=dictionary, passes=2,workers=4)
        file2 = open('files/tfidf_topic_models' + datetime.date.today().strftime("%B %d, %Y") + '.csv', 'w',encoding='utf-8')
        allTopics_tfidf={}
        for idx, topic in lda_model_tfidf.print_topics(-1,15):
            allTopics_tfidf[str(idx)]=dict([(str(dictionary[wid]), str(s)) for (wid, s) in lda_model_tfidf.get_topic_terms(idx)])
            file2.write('Topic: {} \nWords: {}'.format(idx, topic))
            file2.write('\n')
        file2.close()
        pickle.dump(lda_model, open('files/lda_bow', 'wb'))
        pickle.dump(lda_model_tfidf, open('files/lda_tfidf', 'wb'))
        pickle.dump(bow_corpus, open('files/bow_corpus', 'wb'))
        pickle.dump(corpus_tfidf, open('files/tfidf_corpus', 'wb'))
        pickle.dump(processed_docs, open('files/docs', 'wb'))
        print('...topic models done...')
        return allTopics,allTopics_tfidf

    def mention_network(self):
        G = nx.Graph()
        iterator = self.collection.find()
        for i in iterator:
            user = i['user']['id_str']
            entities = i['entities']
            if 'user_mentions' in entities:
                for m in entities['user_mentions']:
                    if G.has_edge(user, m['id_str']):
                        continue
                    else:
                        G.add_edge(user, m['id_str'])
        density = nx.density(G)
        volume = len(G.nodes())
        mass = len(G.edges())
        triangles = sum(nx.triangles(G).values()) / 3
        # diameter = nx.diameter(G)
        nx.write_edgelist(G, "test.csv", delimiter=" ")
        obj={'Graph':G,"density":density,"volume":volume,"mass":mass,"triangles":triangles}
        return obj




if __name__ == "__main__":
    # twts =  get_tweets()
    # pickle.dump(twts,open('test/tweets','wb'))
    # twts = pickle.load(open('test/tweets','rb'))
    # t = TweetAnalysis(col,twts)
    # users = t.users()
    # print (len(set(users)))
    # mau = t.most_active_users(100)[1]
    # mft = t.most_favorited_tweets(10)
    # print (mft)
    # mrt = t.most_retweeted_tweets(10)
    # wf = t.wordFrequenciesCsv()
    # hf = t.hashtagFrequenciesCsv()
    # tmod = t.topic_models(10)
    # tu = t.top_n_urls(15)[1]
    # obj ={'dateTime':datetime.datetime.now(),"total_users":len(set(users)),"most_active_users":mau,"most_fav_tweets":mft[1],
    #       "most_retweeted_tweets":mrt[1],"word_freq":wf,"hashtag_freq":hf,'top_urls':tu,'topics_tfidf':tmod[1],'topics_bow':tmod[0]}
    # pickle.dump(obj,open('analysis_results','wb'))
    # resultsCol.insert_one(obj)
    # t.hashtagCloud()
    # t.textCloud()
    # pprint(t.mention_network())
    import json
    obj = pickle.load(open('analysis_results','rb'))
    with open("sample.json", "w",encoding='utf-8') as outfile:
        json.dump(obj, outfile,sort_keys=True, default=str)
    # for k,v in obj.items():
    #     if k =='topics_tfidf':
    #         print (k,v)
