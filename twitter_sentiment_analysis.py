import re, os, traceback
from unicodedata import normalize
import string
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
from pandas import ExcelWriter
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from many_stop_words import get_stop_words
from collections import defaultdict

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional

from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors


class TwitterSentimentAnalysis(object):
    
    def __init__(self):

        self.embedding_dim = 300
        
        # configurable parameters
        self.max_review_length = 500
        self.top_words = 300000
        self.test_size = 0.3
        self.lemma_flag = 1
        self.max_letters = 2
        self.threshold = 0.5
        self.remove_stopwords = 1
        self.batch_size = 512
        self.epochs = 200

        self.stopwords = stopwords.words('english')
        if self.remove_stopwords: 
            # self.stopWords = list(get_stop_words('en'))
            self.stopwords = []
            print("\n len(self.stopwords) = ", len(self.stopwords))
        self.tokenizer = Tokenizer(num_words=self.top_words)

    
    def readData(self):
        # step 1 === read dataset
        train  = pd.read_csv('train_E6oV3lV.csv')
        test = pd.read_csv('test_tweets_anuFYb8.csv')
        
        print('\n train size == ', len(train), train.keys())
        print('\n test size == ', len(test))
        print("\n summary : \n",train.head())
        return train, test
        

    def clean_str(self, string): # unused
        """
        Tokenization/string cleaning for datasets.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()


    def removeStopWords(self, text):
        return " ".join([token for token in text.split() if token not in self.stopwords])
            

    
    def checkLemma(self, wrd):
        return nltk.stem.WordNetLemmatizer().lemmatize(nltk.stem.WordNetLemmatizer().lemmatize(wrd, 'v'), 'n')

    
    def getLemma(self, text):
        text_list = []
        text_list = [self.checkLemma(tok) for tok in text.lower().split()]
        text = " ".join(text_list)
        return text

    
    ## function to remove twitter handle
    def remove_pattern(self, txt, pattern):
        r = re.findall(pattern, txt)
        for i in r:
            txt = re.sub(i, '', txt)
            txt = re.sub(r"[^A-Za-z0-9]", " ", txt)
            # txt = txt.decode('utf-8')
            # txt = normalize('NFKD', txt).encode('ASCII', 'ignore')
            txt = " ".join([str(word) for word in txt.split() if word not in string.ascii_letters])
        return txt


    def dataPreprocessing(self, train, test):
        ############################################## train
        train['tidy_tweet'] = np.vectorize(self.remove_pattern)(train['tweet'], "@[\w]*")
        train['tidy_tweet'] == train['tidy_tweet'].str.replace("^A-Za-z#", " ")
        train['tidy_tweet'] = train['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>self.max_letters and w not in self.stopwords]))
        train['tidy_tweet'] = [ self.clean_str(sent) for sent in train['tidy_tweet']]

        if self.lemma_flag: train['tidy_tweet'] = [ self.getLemma(sent) for sent in train['tidy_tweet']]
        tokenized_tweet_train = train['tidy_tweet'].apply(lambda x : x.split())   
        
        ############################################## test
        test['tidy_tweet'] = np.vectorize(self.remove_pattern)(test['tweet'], "@[\w]*")
        test['tidy_tweet'] == test['tidy_tweet'].str.replace("^A-Za-z#", " ")
        test['tidy_tweet'] = test['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>self.max_letters and w not in self.stopwords]))
        test['tidy_tweet'] = [ self.clean_str(sent) for sent in test['tidy_tweet']]

        if self.lemma_flag: test['tidy_tweet'] = [ self.getLemma(sent) for sent in test['tidy_tweet']]
        tokenized_tweet_test = test['tidy_tweet'].apply(lambda x : x.split())   
        
        return tokenized_tweet_train, tokenized_tweet_test


    # ste 5 === load word2vec
    def loadw2vLocal(self):
        if not os.path.exists('/home/swapnil/Downloads/NLP/codes/word2vec/GoogleNews-vectors-negative300.bin.gz'):
            raise ValueError('google word2vec model is not there !! ')
        
        model = KeyedVectors.load_word2vec_format('/home/swapnil/Downloads/NLP/codes/word2vec/GoogleNews-vectors-negative300.bin.gz', limit=600000, binary=True)
        return model

    def createEmbeddingLayer(self, word_index, w2vmodel):
        # ste 6 === create embedding matrix
        embedding_matrix = np.zeros((self.top_words, self.embedding_dim))
        
        for word, i in word_index.items():
            if i >= self.top_words:
                continue
            else:
                embedding_vector = np.zeros((1, self.embedding_dim)) # vector of 1 x 300
                try:
                    embedding_vector = w2vmodel[word]
                except:
                    pass
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
        
        print("\n embedding_matrix = ", embedding_matrix.shape)
        embed_layer = Embedding(self.top_words, self.embedding_dim, weights=[embedding_matrix], input_length=self.max_review_length)
        return embed_layer

    def train(self, embed_layer, x_tr, y_tr, x_val, y_val):
        #step 7 ===  create the model
        model = Sequential()
        model.add(embed_layer)
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        # model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(200)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='relu'))
        
        model.compile(loss= 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        model.fit(x_tr, y_tr, epochs=self.epochs, batch_size=self.batch_size)
        scores = model.evaluate(x_val, y_val, verbose=0)
        
        print('\n evaluation accuracy === ',scores[0])
        return model


    def test(self,tokenized_tweet_test, model):
        sequences = self.tokenizer.texts_to_sequences(tokenized_tweet_test)
        x_predict = sequence.pad_sequences(sequences, maxlen=self.max_review_length)
        y_prob = model.predict(x_predict)
        
        test_results = [lst[0] for lst in y_prob]
        test_labels = [0 if score < self.threshold else 1 for score in test_results ]
        
        print("\n len(test_labels) === ",len(test_labels))
        return test_labels


    def writeToCsv(self, test, test_labels):
        try:
            submission = defaultdict(list)
            submission['id'].extend(test['id'])
            submission['label'].extend(test_labels)
        
            print("\n id len === ",len(submission['id']))
            print("\n label === ", len(submission['label']))
        
            submission = pd.DataFrame(submission)
            writer = ExcelWriter('submission_200.xlsx')
            submission.to_excel(writer)
            writer.save()
            print("\n saved results in csv successfully === ")
        
        except Exception as e:
            print("\n error ", e, "\n traceback === ",traceback.format_exc())
    
    
    def main(self):
        
        train, test = self.readData()
        tokenized_tweet_train, tokenized_tweet_test = self.dataPreprocessing(train, test)
        x_tr, x_val, y_tr, y_val = train_test_split( tokenized_tweet_train, train['label'], test_size = self.test_size, random_state=42 )

        self.tokenizer.fit_on_texts(x_tr)
        sequences = self.tokenizer.texts_to_sequences(x_tr)
        word_index = self.tokenizer.word_index
        x_tr = sequence.pad_sequences(sequences, maxlen=self.max_review_length)
        
        sequences = self.tokenizer.texts_to_sequences(x_val)
        x_val = sequence.pad_sequences(sequences, maxlen=self.max_review_length)
        
        w2vmodel = self.loadw2vLocal()

        embed_layer = self.createEmbeddingLayer(word_index, w2vmodel)
        model = self.train(embed_layer, x_tr, y_tr, x_val, y_val)
        test_labels = self.test(tokenized_tweet_test, model)
        self.writeToCsv(test, test_labels)
        
if __name__ == '__main__':
    obj = TwitterSentimentAnalysis()
    obj.main()
    
### improvements
# 1. stopwords
# 2. lemmatize
