import numpy as np
import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
import pickle
import data_utils
import glove_utils
import models
import display_utils
from goog_lm import LM

import lm_data_utils
import lm_utils
from attacks import GeneticAtack
import fasttext
from sklearn.metrics import accuracy_score

np.random.seed(1001)
tf.set_random_seed(1001)


VOCAB_SIZE  = 50000
MAX_VOCAB_SIZE = 50000
with open('aux_files/dataset_%d.pkl' %VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)

doc_len = [len(dataset.test_seqs2[i]) for i in 
           range(len(dataset.test_seqs2))]


max_len = 250
pop_size = 20
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

model = fasttext.load_model("cail_0518/fasttext_model.bin")


y_true = []
y_pred = []

for i in range(len(dataset.test_y)):
    x_orig = test_x[i]
    orig_label = int(test_y[i][0])
    x_orig_i_str = ' '.join([str(a) for a in x_orig])

    idx, probs =  model.predict(x_orig_i_str, k=pop_size)
    idx = [int(item.split('_')[-1]) for item in idx]
    idx = np.argsort(idx)
    orig_preds = probs[idx]    
    pred = np.argmax(orig_preds)
    y_true.append(orig_label)
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
print(y_true.shape)
print(y_pred.shape)
print(accuracy_score(y_true, y_pred))
