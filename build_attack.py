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
from build_complete import build_complete_sentence

np.random.seed(1001)
tf.set_random_seed(1001)

# %load_ext autoreload
# %autoreload 2

VOCAB_SIZE  = 60702
MAX_VOCAB_SIZE = 60702
with open('aux_files/dataset_%d.pkl' %VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)

doc_len = [len(dataset.test_seqs2[i]) for i in 
           range(len(dataset.test_seqs2))]

# embedding_matrix = np.load(('aux_files/embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
# # missed = np.load(('aux_files/missed_embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
# c_ = -2*np.dot(embedding_matrix.T , embedding_matrix)
# a = np.sum(np.square(embedding_matrix), axis=0).reshape((1,-1))
# b = a.T
# dist_mat = a+b+c_
# # dist_mat = np.load('aux_files/dist_counter_%d.npy' %VOCAB_SIZE)
# # Prevent returning 0 as most similar word because it is not part of the dictionary
# dist_mat[0,:] = 100000
# dist_mat[:,0] = 100000

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


SAMPLE_SIZE = 30
TEST_SIZE = 200
test_idx = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)
test_len = []
for i in range(SAMPLE_SIZE):
    test_len.append(len(dataset.test_seqs2[test_idx[i]]))
print('Shortest sentence in our test set is %d words' %np.min(test_len))

test_list = []
orig_list = []
orig_label_list = []
adv_list = []
dist_list = []

for i in range(SAMPLE_SIZE):
    x_orig = test_x[test_idx[i]]

    for w_id in x_orig.flatten().tolist():
        if w_id == 0:
            print()
            break
        print(dataset.inv_dict[w_id], end='\t')


    orig_label = int(test_y[test_idx[i]][0])
    # print(x_orig[np.newaxis,:][0])
    x_orig_i_str = ' '.join([str(a) for a in x_orig])
    # print(x_orig_i_str)
    # print(model.predict([x_orig_i_str]))
    pred_labels, probs =  model.predict(x_orig_i_str)
    print(pred_labels, probs)

    pred_labels = int(pred_labels[0].split('_')[-1])
    print(orig_label, pred_labels)
    if pred_labels != orig_label:
        print('skipping wrong classifed ..')
        print('--------------------------')
        continue
    x_len = np.sum(np.sign(x_orig))
    print(x_len)
    if x_len >= 150:
        print('skipping too long input..')
        print('--------------------------')
        continue
    # if np.max(probs) < 0.90:
    #    print('skipping low confidence .. \n-----\n')
    #    continue
    print('****** ', len(test_list) + 1, ' ********')
    test_list.append(test_idx[i])
    orig_list.append(x_orig)
    target_label = orig_label
    ## delete
    orig_label_list.append(orig_label)
    x_adv = ga_atttack.attack( x_orig, target_label)
    adv_list.append(x_adv)
    if x_adv is None:
        print('%d failed' %(i+1))
        dist_list.append(100000)
    else:
        # print('x_adv:',x_adv)
        num_changes = np.sum(x_orig != x_adv)
        print('%d - %d changed.' %(i+1, num_changes))
        dist_list.append(num_changes)
        for w_id in x_adv.flatten().tolist():
            if w_id == 0:
                print()
                break
            print(dataset.inv_dict[w_id], end='\t')
        x_adv_i_str = ' '.join([str(a) for a in x_adv])
        pred_labels, probs =  model.predict(x_adv_i_str)
        print(pred_labels, probs)
        
        # display_utils.visualize_attack(sess, model, dataset, x_orig, x_adv)
    print('--------------------------')
    if (len(test_list)>= TEST_SIZE):
        break
