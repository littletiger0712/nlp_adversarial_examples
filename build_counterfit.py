"""
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

import os
#import nltk
import re
from collections import Counter


import data_utils
import glove_utils
import json

IMDB_PATH = 'cail_0518'
MAX_VOCAB_SIZE = 50000
GLOVE_PATH = 'cc.zh.300.vec'
train_path = IMDB_PATH + '/data_train_seg.txt'

def not_have_sig(word):
    num_let_sig_list = ['「','|','…','!','%','�','~','1','2','3','4','5','6','7','8','9','0', 'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', 'A','B','C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', '?', '_', '“', '”', '、', '。', '《', '》', '！', '，', '：', '；', '？', '———', '》），', '）÷（１－', '”，', '）、', '＝（', ':', '→', '℃', '&', '*', '一一', '~~~~', '’', '.', '『', '.一', './', '--', '』', '＝″', '【', '［＊］', '｝＞', '［⑤］］', '［①Ｄ］', 'ｃ］', 'ｎｇ昉', '＊', '//', '［', '］', '［②ｅ］', '［②ｇ］', '＝｛', '}', '，也', '‘', 'Ａ', '［①⑥］', '［②Ｂ］', '［①ａ］', '［④ａ］', '［①③］', '［③ｈ］', '③］', '１．', '－－', '［②ｂ］', '’‘', '×××', '［①⑧］', '０：２', '＝［', '［⑤ｂ］', '［②ｃ］', '［④ｂ］', '［②③］', '［③ａ］', '［④ｃ］', '［①⑤］', '［①⑦］', '［①ｇ］', '∈［', '［①⑨］', '［①④］', '［①ｃ］', '［②ｆ］', '［②⑧］', '［②①］', '［①Ｃ］', '［③ｃ］', '［③ｇ］', '［②⑤］', '［②②］', '一.', '［①ｈ］', '.数', '［］', '［①Ｂ］', '数/', '［①ｉ］', '［③ｅ］', '［①①］', '［④ｄ］', '［④ｅ］', '［③ｂ］', '［⑤ａ］', '［①Ａ］', '［②⑦］', '［①ｄ］', '［②ｊ］', '〕〔', '］［', '://', '′∈', '［②④', '［⑤ｅ］', '１２％', 'ｂ］', '...', '...................', '…………………………………………………③', 'ＺＸＦＩＴＬ', '［③Ｆ］', '」', '［①ｏ］', '］∧′＝［', '∪φ∈', '′｜', '｛－', '②ｃ', '｝', '［③①］', 'Ｒ．Ｌ．', '［①Ｅ］', 'Ψ', '－［＊］－', '↑', '.日', '［②ｄ］', '［②', '［①②］', '［②ａ］', 'ｆ］', '［⑩］', 'ａ］', '［①ｅ］', '［②ｈ］', '［②⑥］', '［③ｄ］', '［②⑩］', 'ｅ］', '〉', '】', '元／吨', '２．３％', '５：０', '［①］', '::', '［②］', '［③］', '［④］', '［⑤］', '［⑥］', '［⑦］', '［⑧］', '［⑨］', '……', '——', '．', ',', '\'','·', '──', '—', '<', '>', '（', '）', '〔', '〕', '[', ']', '(', ')', '-', '+', '～', '×', '／', '/', '①', '②', '③', '④', '⑤', '⑥', '⑦', '⑧', '⑨', '⑩', 'Ⅲ', 'В', '"', ';', '#', '@', 'γ', 'μ', 'φ', 'φ．', 'Δ', '■', '▲', 'sub', 'exp', 'sup', 'Lex', '＃', '％', '＆', '＇', '＋', '＋ξ', '＋＋', '－', '－β', '＜', '＜±', '＜Δ', '＜λ', '＜φ', '＜＜', '=', '＝', '＝☆', '＝－', '＞', '＞λ', '＿', '～±', '～＋', '［⑤ｆ］', '［⑤ｄ］', '［②ｉ］', '≈', '［②Ｇ］', '［①ｆ］', 'ＬＩ', '㈧', '［－', '......']
    for item in num_let_sig_list:
        if item in word:
            return False
    return True
        

def read_text(path):
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


vec_file_name = 'cc.zh.300.vec'
vec_file = open(vec_file_name,'r')
stop_words_file = open('cail_0518/stopwords.txt','r')
stop_list = []
for line in stop_words_file:
    stop_list.append(line.strip())
print(stop_list)
vec_list = []
cnt = 0
for line in vec_file:
    vec = line.strip().split()[0]
    if vec not in stop_list and not_have_sig(vec):
        vec_list.append(vec)
        cnt += 1
    if cnt == 50000:
        break
print(vec_list)


train_text, _ = read_text(train_path)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_text)
x = tokenizer.word_counts
word_counts_list = []
for w,i in x.items():
    if i > 20:
        word_counts_list.append(w)
# word_counts_list = [w for w in x if int(x[w]) > 20]
print(word_counts_list)


# print(len(vec_list))
# print(len(word_counts_list))
all_list = list(set(vec_list + word_counts_list))
# print(all_list)
# print(len(all_list))
all_str = ' '.join(all_list)
out_file = open('dictionary_6w.txt','w')
out_file.write(all_str)




# imdb_dataset = data_utils.IMDBDataset(path=IMDB_PATH, max_vocab_size=MAX_VOCAB_SIZE)
# print(imdb_dataset.dict)

# glove = glove_utils.loadGloveModel(GLOVE_PATH)
# glove_dist_nearest = {}
# for w_cur,i_cur in glove.items():
#     glove_dist = {}
#     for w,i in glove.items():
#         glove_dist[w] = np.linalg.norm(i - i_cur)
#     glove_dist_nearest[w_cur] = sorted(glove_dist.items(), key = lambda kv:(kv[1],kv[0]))[1:51]

# for w,i in imdb_dataset.dict:

# out_file_name = 'counter-fitted-vectors-nearest50.txt'
# out_file = open(out_file_name,'w')
# out_file.write(json.dumps(glove_dist_nearest))
# out_file.close()
# -------------------------------------------------------------------------------   


# def create_counterfit(glove_model, dictionary):
#     # MAX_VOCAB_SIZE = len(dictionary)
#     # Matrix size is 300
#     dic = {}
#     for w, i in dictionary.items():
#         if w in glove_model:
#             dic[w] = glove_model[w]
#     return dic

# if not os.path.exists('aux_files'):
# 	os.mkdir('aux_files')
# imdb_dataset = data_utils.IMDBDataset(path=IMDB_PATH, max_vocab_size=MAX_VOCAB_SIZE)

# # save the dataset
# # with open(('aux_files/dataset_%d.pkl' %(MAX_VOCAB_SIZE)), 'wb') as f:
# #     pickle.dump(imdb_dataset, f)


# # Load the counterfitted-vectors (used by our attack)
# glove2 = glove_utils.loadGloveModel(GLOVE_PATH)
# # create embeddings matrix for our vocabulary
# counter_embeddings = create_counterfit(glove2, imdb_dataset.dict)

# # save the embeddings for both words we have found, and words that we missed.
# out_file_name = 'counter-fitted-vectors-chinese.txt'
# out_file = open(out_file_name,'w')
# for w,i in counter_embeddings.items():
#     # print(w)
#     # print(i)
#     list_i = [str(item) for item in list(i)]
#     out_file.write(w + ' ' + ' '.join(list_i)+ '\n')