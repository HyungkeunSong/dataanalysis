
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3


# In[2]:


from __future__ import print_function


# In[3]:


titles = open('/Users/hyungkeun/Desktop/pythondata/title_list2.txt').read().split('\n')


# In[4]:


len(titles)


# In[5]:


titles[:10]


# In[6]:


contents = open('/Users/hyungkeun/Desktop/pythondata/contents2.txt').read().split('\n')


# In[7]:


len(contents)


# In[8]:


contents[0]


# In[9]:


# load nltk's English stopwords as variable called 'stopwords'
# use nltk.download() to install the corpus first
# Stop Words are words which do not contain important significance to be used in Search Queries
stopwords = nltk.corpus.stopwords.words('english')


# In[10]:


# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")


# In[11]:


# In[21]:

len(stopwords)


# In[12]:


stopwords


# In[13]:


from nltk.stem.wordnet import WordNetLemmatizer


# In[14]:


lem = WordNetLemmatizer() #text10 = [lem.lemmatize(text8, "v")]


# In[15]:


# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed
# Punkt Sentence Tokenizer, sent means sentence 
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[16]:


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


# In[18]:


totalvocab_stemmed = []
totalvocab_tokenized = []


# In[19]:


for i in contents:
    allwords_stemmed = tokenize_and_stem(i) # for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) # extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


# In[20]:


print(len(totalvocab_stemmed))
print(len(totalvocab_tokenized))


# In[21]:


totalvocab_stemmed


# In[22]:


totalvocab_stemmed = [lem.lemmatize(t, "v") for t in totalvocab_stemmed]


# In[23]:


vocab_frame = pd.DataFrame({'words': totalvocab_stemmed}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame.head())


# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[25]:


tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,2))


# In[26]:


get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(contents) #fit the vectorizer to contents')


# In[27]:


print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
len(terms)


# In[28]:


from sklearn.metrics.pairwise import cosine_similarity
# A short example using the sentences above
words_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,2))


# In[29]:


from sklearn.cluster import KMeans


# In[30]:


num_clusters = 5

km = KMeans(n_clusters=num_clusters)

get_ipython().magic(u'time km.fit(tfidf_matrix)')

clusters = km.labels_.tolist()


# In[31]:


from sklearn.externals import joblib


# In[32]:


joblib.dump(km, 'doc_cluster.pkl')


# In[33]:


km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
# clusters show which cluster (0-4) each of the 100 synoposes belongs to
print(len(clusters))
print(clusters)


# In[34]:


updates = {'title': titles, 'contents': contents, 'cluster': clusters}


# In[35]:


frame = pd.DataFrame(updates, index = [clusters] , columns = ['title', 'cluster'])


# In[36]:

# frame.to_excel("/Users/hyungkeun/Desktop/pythondata/cluster_num5.xls")


# In[37]:


frame


# In[38]:


frame['cluster'].value_counts()


# In[39]:


from __future__ import print_function


# In[41]:


print("Top terms per cluster:")

#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.loc[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print() #add whitespace
    print() #add whitespace


# In[42]:


similarity_distance = 1 - cosine_similarity(tfidf_matrix)
print(type(similarity_distance))
print(similarity_distance.shape)


# In[43]:


print(similarity_distance)


# In[44]:


import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS


# In[45]:


# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)


# In[46]:


get_ipython().magic(u'time pos = mds.fit_transform(similarity_distance)  # shape (n_components, n_samples)')


# In[47]:


print(pos.shape)
print(pos)


# In[48]:


xs, ys = pos[:, 0], pos[:, 1]
print(type(xs))
xs


# In[49]:


cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: 'gray'}

#set up cluster names using a dict
cluster_names = {0: 'Cluster1', 
                 1: 'Cluster2', 
                 2: 'Cluster3',
                 3: 'Cluster4',
                 4: 'Cluster5',
                 5: 'Cluster6'}


# In[50]:


get_ipython().magic(u'matplotlib inline')


# In[51]:


#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 


# In[52]:


print(df[1:10])
# group by cluster
# this generate {name:group(which is a dataframe)}
groups = df.groupby('label')
print(groups.groups)


# In[53]:


import plotly.plotly as py


# In[54]:


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
# ms: marker size
for name, group in groups:
    print("*******")
    print("group name " + str(name))
    print(group)
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='False',      # ticks along the bottom edge are off
        top='False',         # ticks along the top edge are off
        labelbottom='False')
    ax.tick_params(        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='False',      # ticks along the bottom edge are off
        top='False',         # ticks along the top edge are off
        labelleft='False')
    
ax.legend(numpoints=1)  #show legend with only 1 point


#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=12)  

    
plt.show() #show the plot

