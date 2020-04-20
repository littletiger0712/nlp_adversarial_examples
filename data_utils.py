"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import os
#import nltk
import re
from collections import Counter
#from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import pickle as pickle

class IMDBDataset(object):
    def __init__(self, path='aclImdb', max_vocab_size=None):
        self.path = path
        self.train_path = path + '/data_train_seg.txt'
        self.test_path = path + '/data_test_seg.txt'
        self.vocab_path = path + '/imdb.vocab'
        self.token_words_path = 'dictionary_6w.txt'
        self.max_vocab_size = max_vocab_size
        # self._read_vocab()
        train_text, self.train_y = self.read_text(self.train_path)
        test_text, self.test_y = self.read_text(self.test_path)
        self.train_text = train_text
        self.test_text = test_text
        # print(self.train_text)
        print('tokenizing...')
        
        # Tokenized text of training data
        self.tokenizer = Tokenizer()
        token_words_file = open(self.token_words_path,'r')
        token_words_list = [line for line in token_words_file]
        
        #nlp = spacy.load('en')
        # train_text = [nltk.word_tokenize(doc) for doc in train_text]
        # test_text = [nltk.word_tokenize(doc) for doc in test_text]
        #train_text = [[w.string.strip() for w in nlp(doc)] for doc in train_text]
        #test_text = [[w.string.strip() for w in nlp(doc)] for doc in test_text]
        self.tokenizer.fit_on_texts(token_words_list)
        # print(self.tokenizer.word_index)
        if max_vocab_size is None:
            max_vocab_size = len(self.tokenizer.word_index) + 1
        #sorted_words = sorted([x for x in self.tokenizer.word_counts])
        #self.top_words = sorted_words[:max_vocab_size-1]
        #self.other_words = sorted_words[max_vocab_size-1:]
        self.dict = dict()
        self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
        self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]
        
        self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
        self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]
        
        self.dict['UNK'] = max_vocab_size
        self.inv_dict = dict()
        self.inv_dict[max_vocab_size] = 'UNK'
        self.full_dict = dict()
        self.inv_full_dict = dict()
        for word, idx in self.tokenizer.word_index.items():
            if idx < max_vocab_size:
                self.inv_dict[idx] = word
                self.dict[word] = idx
            self.full_dict[word] = idx
            self.inv_full_dict[idx] = word 
        print('Dataset built !')
        
    def save(self, path='imdb'):
        with open(path + '_train_set.pickle', 'wb') as f:
            pickle.dump((self.train_text, self.train_seqs, self.train_y) , f)

        with open(path + '_test_set.pickle', 'wb') as f:
            pickle.dump((self.test_text, self.test_seqs, self.test_y) , f)

        with open(path + '_dictionary.pickle', 'wb') as f:
            pickle.dump((self.dict, self.inv_dict) , f)

    def _read_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab_words = f.read().split('\n')
            self.vocab = dict([(w,i) for i, w in enumerate(vocab_words)])
            self.reverse_vocab = dict([(i,w) for w,i in self.vocab.items()])

    def read_text(self, path):
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        pos_list = []
        labels_list = []
        pos_file = open(path,'r')
        # pos_files = [path + '/' + x for x in os.listdir(path) if x.endswith('.txt')]
        for line in pos_file:
            line_list = line.strip().split('\t')
            sentence = line_list[0]
            pos_list.append(sentence)
            labels_list.append(line_list[1].split('&'))
        return pos_list, labels_list
    
    def build_text(self, text_seq):
        text_words = [self.inv_full_dict[x] for x in text_seq]
        return ' '.join(text_words)



if __name__ == "__main__":
    IMDB_PATH = 'cail_0518'
    # MAX_VOCAB_SIZE = 50000
    imdb_dataset = IMDBDataset(path=IMDB_PATH)
    # print(imdb_dataset.train_seqs)
    # print(len(imdb_dataset.train_seqs))
    # print(len(imdb_dataset.full_dict))
    print(imdb_dataset.dict)
    print(imdb_dataset.inv_dict)