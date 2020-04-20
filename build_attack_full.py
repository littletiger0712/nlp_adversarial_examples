import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
import pickle
import data_utils
import glove_utils
import models
# import display_utils
from goog_lm import LM

import lm_data_utils
import lm_utils
from attacks import GeneticAtack
import fasttext
import copy
import json
# from build_complete import build_complete_sentence

np.random.seed(1001)
tf.set_random_seed(1001)

# %load_ext autoreload
# %autoreload 2



def write_json_file(output, doc):
    with open(output, 'w', encoding='utf-8')as fw:
        json.dump(doc, fw, ensure_ascii=False)
        # fw.write(doc)
    fw.close()


VOCAB_SIZE  = 60702
MAX_VOCAB_SIZE = 60702
with open('aux_files/dataset_%d.pkl' %VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)

doc_len = [len(dataset.test_seqs2[i]) for i in
           range(len(dataset.test_seqs2))]


dist_mat_list = np.load('aux_files/sdist_mat_dic_%d.npy' % (MAX_VOCAB_SIZE))
dist_mat_order = np.load('aux_files/sdist_order_%d.npy' % (MAX_VOCAB_SIZE))


skip_list = np.load('aux_files/missed_embeddings_counter_%d.npy' %VOCAB_SIZE)

for i in range(20, 40):
    src_word = i
    nearest, nearest_dist = glove_utils.pick_most_similar_words(src_word, dist_mat_order, dist_mat_list, 10)

    print('Closest to `%s` are:' %(dataset.inv_dict[src_word]))
    for w_id, w_dist in zip(nearest, nearest_dist):
          print(' -- ', dataset.inv_dict[w_id], ' ', w_dist)

    print('----')

# Preparing the dataset
max_len = 250
# train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
# train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)


model = fasttext.load_model("cail_0518/fasttext_model.bin")

pop_size = 20
n1 = 10
ga_atttack = GeneticAtack(model, model, model, dataset, dist_mat_order, dist_mat_list,
                                  skip_list,
                                  None, max_iters=1000,
                                   pop_size=pop_size,
                                  n1 = n1,
                                  n2 = 4,
                                 use_lm = False, use_suffix=False)


# SAMPLE_SIZE = 30
# TEST_SIZE = 200
# test_idx = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)
# test_len = []
# for i in range(SAMPLE_SIZE):
#     test_len.append(len(dataset.test_seqs2[test_idx[i]]))
# print('Shortest sentence in our test set is %d words' %np.min(test_len))

test_list = []#被对抗攻击的test序号
orig_list = []#被对抗攻击的test原数据（序号）
orig_word_list = []#被对抗攻击的test原数据(文字)
orig_label_list = []#被对抗攻击的test原始label
orig_label_list_zhixin = []#被对抗攻击的test原始label置信度
adv_list = []#经过对抗攻击的文本(序号)
adv_word_list = []#经过对抗攻击的文本（文字）
adv_label_list = []#被对抗攻击的test对抗后label
adv_label_list_zhixin = []#被对抗攻击的test对抗后label置信度
dist_list = []#改变的词语数量

TEST_NUM = 10
test_text_file = open('cail_0518/data_test_text.txt','r')
test_text = []#未经过分词的test文本
cnt = 0
for line in test_text_file:
    if cnt == TEST_NUM:
        break
    test_text.append(line.strip())
    cnt += 1
test_text_file.close()


for i in range(TEST_NUM):
    x_orig = test_x[i]
    # print(x_orig)
    word_list = []
    for w_id in x_orig.flatten().tolist():
        if w_id == 0:
            print()
            break
        print(dataset.inv_dict[w_id], end='\t')
        word_list.append(dataset.inv_dict[w_id])


    orig_label = int(test_y[i][0])
    # print(x_orig[np.newaxis,:][0])
    x_orig_i_str = ' '.join([str(a) for a in x_orig])
    # print(x_orig_i_str)
    # print(model.predict([x_orig_i_str]))
    pred_labels, probs =  model.predict(x_orig_i_str)
    print(pred_labels, probs)

    pred_labels = int(pred_labels[0].split('_')[-1])
    if pred_labels != orig_label:
        print('skipping wrong classifed ..')
        print('--------------------------')
        continue
    x_len = np.sum(np.sign(x_orig))

    print('****** ', len(test_list) + 1, ' ********')
    test_list.append(i)
    orig_list.append(x_orig)
    orig_word_list.append(word_list)
    target_label = orig_label
    ## delete
    orig_label_list.append(orig_label)
    orig_label_list_zhixin.append(probs)
    x_adv = ga_atttack.attack(x_orig, target_label)
    if x_adv is None:
        adv_list.append('None')
    adv_list.append(x_adv)
    if x_adv is None:
        print('%d failed' %(i+1))
        dist_list.append(100000)
        adv_word_list.append('None')
        adv_label_list.append('None')
        adv_label_list_zhixin.append('None')
    else:
        # print('x_adv:',x_adv)
        num_changes = np.sum(x_orig != x_adv)
        print('%d - %d changed.' %(i+1, num_changes))
        dist_list.append(num_changes)
        word_list2 = []
        for w_id in x_adv.flatten().tolist():
            if w_id == 0:
                print()
                break
            print(dataset.inv_dict[w_id], end='\t')
            word_list2.append(dataset.inv_dict[w_id])
        # print(build_complete_sentence(exchange_list[i], word_list))
        adv_word_list.append(word_list2)
        x_adv_i_str = ' '.join([str(a) for a in x_adv])
        pred_labels, probs =  model.predict(x_adv_i_str)
        adv_label_list.append(pred_labels)
        adv_label_list_zhixin.append(probs)
        print(pred_labels, probs)

        # display_utils.visualize_attack(sess, model, dataset, x_orig, x_adv)
    print('--------------------------')
    # if (len(test_list)>= TEST_SIZE):
    #     break
print('test_list:',test_list)
print('orig_list:',orig_word_list)
print('orig_label_list:',orig_label_list)
print('orig_label_list_zhixin:',orig_label_list_zhixin)
print('adv_list:',adv_word_list)
print('adv_label_list:',adv_label_list)
print('adv_label_list_zhixin:',adv_label_list_zhixin)
print('dist_list',dist_list)

duikang_sucsess = []
cnt = 0
for i in range(len(test_list)):
    if adv_list[i] != 'None':
        dist = int(dist_list[i])
        sentence_len = len(orig_word_list[i])
        tmp = dist / sentence_len
        cnt += tmp
        duikang_sucsess.append(i)
    else:
        pass
avg_change = cnt/TEST_NUM
# print('cnt:',cnt/TEST_NUM)
print('duikang_sucsess:',len(duikang_sucsess))



# def get_complete(com_sentence, ori_list, adv_list):
#     try:
#         for i in range(len(ori_list)):
#             com_sentence = com_sentence.replace(ori_list[i],adv_list[i])
#     except:
#         return com_sentence
#     else:
#         return com_sentence



# for i in range(len(test_list)):
#     test_text_sentence = test_text[test_list[i]]
#     print(test_text_sentence)
#     print(get_complete(test_text_sentence, orig_word_list[i], adv_word_list[i]))



# show_json_list = []
# zuiming_all = ['盗窃',
# '走私、贩卖、运输、制造毒品',
# '故意伤害',
# '抢劫',
# '诈骗',
# '受贿',
# '寻衅滋事',
# '危险驾驶',
# '组织、强迫、引诱、容留、介绍卖淫',
# '制造、贩卖、传播淫秽物品',
# '容留他人吸毒',
# '交通肇事',
# '贪污',
# '非法持有、私藏枪支、弹药',
# '故意杀人',
# '开设赌场',
# '非法持有毒品',
# '职务侵占',
# '强奸',
# '敲诈勒索']


# for i in range(len(test_list)):
#     test_text_sentence = test_text[test_list[i]]
#     test_text_label = orig_label_list[i]
#     print(test_text_sentence)
#     print(test_text_label)
#     show_json_list.append({"id":i,"name":test_text_sentence,"label":zuiming_all[int(test_text_label)]})
#     adv_text_sentence = get_complete(test_text_sentence, orig_word_list[i], adv_word_list[i])
#     adv_text_label = adv_label_list[i][0].split('_')[-1]
#     print(adv_text_sentence)
#     print('!!!',adv_text_label)
#     if adv_text_label != 'N':
#         show_json_list.append({"id":i,"content":adv_text_sentence,"adv_label":zuiming_all[int(adv_text_label)],"change_num":str(dist_list[i])})
#     else:
#         show_json_list.append({"id":i,"content":adv_text_sentence,"adv_label":'None',"change_num":str(dist_list[i])})


# # print(show_json_list)

# show_json = {'a':show_json_list}
# show_json['avg_change'] = avg_change
# show_json['duikang_sucsess'] = len(duikang_sucsess)
# show_json['input_num'] = len(test_list)
# print(show_json)

# abspath = '/home/lw/workspace/zs/webdrive/media/show_attack_1000.json'
# write_json_file(abspath,show_json)
