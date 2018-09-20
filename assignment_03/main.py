import os
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import pandas as pd
from tqdm import tqdm
tqdm.monitor_interval = 0
tqdm.pandas(desc="my bar!")

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
#import nltk
#nltk.download('stopwords')
#nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from string import punctuation
from collections import Counter
from functools import reduce
from wordcloud import WordCloud

from keras.models import Sequential, Model, model_from_json,model_from_yaml
from keras.layers import Input, Dense, Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adagrad, Adadelta, Adamax

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def clean_text(text):
    text = text.lower() 
    text = re.sub(r"what's", "what is ", text)
    text = text.replace('(ap)', '')
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)
    text = re.sub('[^a-zA-Z ?!]+', '', text)
    text = _removeNonAscii(text)
    text = text.strip()
    return text

def tokenizer(text):
    stop_words = []
    f = open('./stopwords.txt', 'r')
    for l in f.readlines():
        stop_words.append(l.replace('\n', ''))       
    text = clean_text(text)    
    tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
    tokens = list(reduce(lambda x,y: x+y, tokens))
    tokens = list(filter(lambda token: token not in (stop_words + list(punctuation)) , tokens))
    return tokens

def plot_word_cloud(terms):
    text = terms.index
    text = ' '.join(list(text))
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


#%% EXTRACT FEATURES

csv_file = os.path.join(os.getcwd(), 'news_headlines.csv')  
year_select = [2017]

data = pd.read_csv(csv_file)

data.head(3)
data.headline_text.map(len).hist(figsize=(15, 5), bins=100)
plt.ylabel('Number of Headlines')
plt.xlabel('Word length')
plt.tight_layout()


print('\n Extracting years of headlines')
data['year'] = data['publish_date'].progress_map(lambda d: datetime.strptime(str(d), "%Y%m%d").year)


print('\n Extracting headlines tokens')
data_yrs = data.loc[data['year'].isin(year_select)]
data_yrs = data_yrs.reset_index()
data_yrs['tokens'] = data_yrs['headline_text'].progress_map(lambda d: tokenizer(d))
data_yrs.head(3)

for descripition, tokens in zip(data_yrs['headline_text'].head(5), data_yrs['tokens'].head(5)):
    print('description:', descripition)
    print('tokens:', tokens)
    print() 


plt.figure()
data_yrs['tokens'].map(len).hist()
plt.ylabel('')
plt.xlabel('Number of Words')
plt.title('Histogram Headlines')
plt.tight_layout()


def countCommonWords(tokens):
    alltokens = []
    for token_list in tokens:
        alltokens += token_list
    return Counter(alltokens)


data_yrs['tokens_original'] = data_yrs['tokens']

counter = countCommonWords(data_yrs['tokens'])
print('top 10 keywords:', counter.most_common(10))

removal_tokens = list (filter(lambda x: x[1] >= 0.01*len(data_yrs), counter.most_common()))
removal_tokens = [x[0] for x in removal_tokens]

def remove_common (listTokens):
    return list(filter(lambda token: token not in removal_tokens, listTokens))
    
data_yrs['tokens'] = data_yrs['tokens'].progress_map(lambda d: remove_common(d))

print('\ntop 10 keywords:', countCommonWords(data_yrs['tokens']).most_common(10))


#%% TFIDF

vectorizer = TfidfVectorizer(min_df=5, analyzer='word', ngram_range=(1, 5), stop_words='english')
vz = vectorizer.fit_transform(list(data_yrs['tokens'].map(lambda tokens: ' '.join(tokens))))
vz.shape

#vectorizer_char = TfidfVectorizer(min_df=5, analyzer='char', ngram_range=(4, 4), stop_words='english')
#vz_char = vectorizer_char.fit_transform(list(data_yrs['tokens'].map(lambda tokens: ' '.join(tokens))))
#vz_char.shape
#
#vectorizer = vectorizer_char
#vz = vz_char


tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
tfidf.columns = ['tfidf']
#tfidf.tfidf.hist(bins=25, figsize=(15,7))
#plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=True).head(40))
#plot_word_cloud(tfidf.sort_values(by=['tfidf'], ascending=False).head(40))

encoded_vz = vz

#%% EMBEDDED

def load_vectors_GLOVE(fname = 'glove.6B/glove.6B.50d.txt'):
    import io
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data_embedding = pd.DataFrame(columns=['word','vec'])
    for idx,line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in counter.keys():
            data_embedding.loc[idx] = [tokens[0], np.array(list(map(float, tokens[1:])))]
    return data_embedding.reset_index()

def load_vectors(fname = 'wiki-news-300d-1M.vec'):
    import io
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data_embedding = pd.DataFrame(columns=['word','vec'])
    for idx,line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in counter.keys():
            data_embedding.loc[idx] = [tokens[0], np.array(list(map(float, tokens[1:])))]
    return data_embedding.reset_index()

#word_embedding = load_vectors()
word_embedding = load_vectors_GLOVE()


vectorizer = CountVectorizer(min_df=5, analyzer='word', stop_words='english', vocabulary= word_embedding['word'])
vz = vectorizer.fit_transform(list(data_yrs['tokens'].map(lambda tokens: ' '.join(tokens))))
vz.shape  # number of headings , number of different words

embedded_matrix = np.array([word_embedding['vec'][i] for i in range (word_embedding.shape[0])])
vz = vz.dot( embedded_matrix )

encoded_vz = vz



#%% LDA

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import gensim.corpora as corpora
from gensim import matutils
from gensim.models import CoherenceModel

aux = data_yrs.copy()

bigram = gensim.models.Phrases(aux['tokens'], min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
aux['tokens_bigram'] = aux['tokens'].progress_map(lambda tokens: bigram_mod[tokens])

id2word = corpora.Dictionary(aux['tokens_bigram'])
texts = aux['tokens_bigram'].values
corpus = [id2word.doc2bow(text) for text in texts]

def LDA_model(num_topics, passes=1):
    return gensim.models.ldamodel.LdaModel(corpus=tqdm(corpus, leave=False),
                                               id2word=id2word,
                                               num_topics=num_topics, 
                                               random_state=100,
                                               eval_every=10,
                                               chunksize=2000,
                                               passes=passes,
                                               per_word_topics=True)

def compute_coherence(model):
    coherence = CoherenceModel(model=model, 
                           texts=aux['tokens_bigram'].values,
                           dictionary=id2word, coherence='c_v')
    return coherence.get_coherence()

def display_topics(model):
    topics = model.show_topics(num_topics=model.num_topics, formatted=False, num_words=10)
    topics = list(map(lambda c: list(map(lambda cc: cc[0], c[1])), topics))
    df = pd.DataFrame(topics)
    df.index = ['topic_{0}'.format(i) for i in range(model.num_topics)]
    df.columns = ['keyword_{0}'.format(i) for i in range(1, 10+1)]
    return df

def explore_models(df, rg=range(5, 25)):
    id2word = corpora.Dictionary(df['tokens_bigram'])
    texts = df['tokens_bigram'].values
    corpus = [id2word.doc2bow(text) for text in texts]

    models = []
    coherences = []
    
    for num_topics in tqdm(rg, leave=False):
        lda_model = LDA_model(num_topics, passes=5)
        models.append(lda_model)
        coherence = compute_coherence(lda_model)
        coherences.append(coherence)
    fig = plt.figure(figsize=(15, 5))
    plt.title('Choosing the optimal number of topics')
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence')
    plt.grid(True)
    plt.plot(rg, coherences)  
    return coherences, models


coherences, models = explore_models(aux, rg=range(5, 85, 5))


best_model = LDA_model(num_topics=40, passes=5)
display_topics(model=best_model)


#%% AUTO-ENCODING

input_dim = vz.shape[1]
encoding_dim = 400

input_img = Input(shape=(input_dim,))
encoded_hidden1 = Dense(1000, activation='sigmoid')(input_img)
encoded_hidden2 = Dense(500, activation='sigmoid')(encoded_hidden1)
encoded = Dense(encoding_dim, activation='linear')(encoded_hidden2)
decoded_hidden2 = Dense(500, activation='sigmoid')(encoded)
decoded_hidden1 = Dense(1000, activation='sigmoid')(decoded_hidden2)
decoded = Dense(input_dim, activation='linear')(decoded_hidden1)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

epochs = 20
import random
for i in range(20):
    idxs_select = random.sample(range(vz.shape[0]),10000)
    autoencoder_train = autoencoder.fit(vz[idxs_select,:], vz[idxs_select,:],
                                        epochs=epochs,
                                        batch_size=1000,
                                        shuffle=True,
                                        validation_split=0.2,
                                        callbacks=[earlyStopping])

encoded_vz = encoder.predict(vz)

vz_reconstr = autoencoder.predict(vz)
vz - vz_reconstr


loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



#%% SVD

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=0)
svd_tfidf = svd.fit_transform(vz)

svd_tfidf.shape

encoded_vz = svd_tfidf

svd_MAE = np.sum(np.sum(np.abs(vz - svd.inverse_transform(encoded_vz)), axis=1)/vz.shape[1])



#%% T-SNE

run = True
if run:
# run this (takes times)
    from sklearn.manifold import TSNE
    import random
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, n_iter=1000)
    idxs_select = random.sample(range(encoded_vz.shape[0]),2000)
    tsne_tfidf = tsne_model.fit_transform(encoded_vz[idxs_select,:])
    print(tsne_tfidf.shape)
    
    tsne_tfidf_df = pd.DataFrame(tsne_tfidf)
    tsne_tfidf_df.columns = ['x', 'y']
    tsne_tfidf_df['headline_text'] = data_yrs.iloc[idxs_select].reset_index()['headline_text'].astype(str)
    tsne_tfidf_df.to_csv('./data/tsne_tfidf.csv', encoding='utf-8', index=False)
else:
# or import the dataset directly
    tsne_tfidf_df = pd.read_csv('./data/tsne_tfidf.csv')


#plt.scatter(tsne_tfidf_df['x'], tsne_tfidf_df['y'])

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file

plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tf-idf clustering of the news",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

plot_tfidf.scatter(x='x', y='y', source=tsne_tfidf_df)
hover = plot_tfidf.select(dict(type=HoverTool))
hover.tooltips={"headline_text": "@headline_text"}

show(plot_tfidf)



#%% KMeans

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from tqdm import tqdm

distorsions = []
k_max = 40
for k in tqdm(range(2, k_max)):
    kmeans_model = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1, random_state=42,  
                         init_size=1000, verbose=False, max_iter=1000)
    kmeans_model.fit(encoded_vz)
    distorsions.append(kmeans_model.inertia_)
    
f, ax1 = plt.subplots(1, 1, sharex=True, figsize=(15, 10))

ax1.plot(range(2, k_max), distorsions)
ax1.set_title('Distorsion vs num of clusters')
ax1.grid(True)
f.tight_layout()


#%% KMeans Champion

num_clusters = 40
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1, random_state=42,                       
                         init_size=1000, batch_size=1000, verbose=False, max_iter=1000, )
kmeans = kmeans_model.fit(encoded_vz)
kmeans_clusters = kmeans.predict(encoded_vz)
kmeans_distances = kmeans.transform(encoded_vz)



#%% TSNE PLOT WITH CLUSTER
import random

for i in range (num_clusters):
    # Select all indexes equal to i
    indexes = np.where (kmeans_clusters == i)
    tokens = [data_yrs['tokens'][j] for j in indexes[0]]

    alltokens = []
    for token_list in tokens:
        alltokens += token_list
    counter = Counter(alltokens)
    print(len(indexes[0]), counter.most_common (10))
    
    data_yrs.head(3)
    idx = random.sample(list(indexes[0]),3)
    data_yrs['headline_text'].iloc[idx]
    for x in data_yrs['headline_text'].iloc[idx].tolist():
        print('\t'+x)
    print('')




run = True
if run:
    from sklearn.manifold import TSNE    
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, n_iter=500)
    idxs_select = random.sample(range(encoded_vz.shape[0]),2000)
    tsne_kmeans = tsne_model.fit_transform(encoded_vz[idxs_select,:])
    kmeans_df = pd.DataFrame(tsne_kmeans, columns=['x', 'y'])
    kmeans_df['cluster'] = kmeans_clusters [idxs_select]
    kmeans_df['cluster'] = kmeans_df['cluster'].map(str)
    kmeans_df['headline_text'] = data_yrs.iloc[idxs_select].reset_index()['headline_text'].astype(str)
    kmeans_df.to_csv('./data/tsne_kmeans.csv', index=False, encoding='utf-8')
else:
    kmeans_df = pd.read_csv('./data/tsne_kmeans.csv')
    kmeans_df['cluster'] = kmeans_df['cluster'].map(str)

#plt.scatter(tsne_tfidf_df['x'], tsne_tfidf_df['y'])

import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook, reset_output
from bokeh.palettes import d3
import bokeh.models as bmo
from bokeh.io import save, output_file


plot_kmeans = bp.figure(plot_width=400, plot_height=300, title="KMeans clustering of the news",
    tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
    x_axis_type=None, y_axis_type=None, min_border=1)

palette = d3['Category20'][20] + d3['Category20b'][20] 
color_map = bmo.CategoricalColorMapper(factors=kmeans_df['cluster'].unique(), palette=palette)

plot_kmeans.scatter('x', 'y', source=kmeans_df, 
                    color={'field': 'cluster', 'transform': color_map}, 
                    legend=False)
hover = plot_kmeans.select(dict(type=HoverTool))
hover.tooltips={"headline_text": "@headline_text", "cluster": "@cluster"}

show(plot_kmeans)

