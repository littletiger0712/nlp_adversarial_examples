import fasttext
import pickle
import numpy as np
max_len = 250
MAX_VOCAB_SIZE = 60702

from keras.preprocessing.sequence import pad_sequences
# file_train_name = 'cail_0518/data_train_seg.txt'
out_file_train_name = 'cail_0518/data_train_fasttext_num.txt'
# file_test_name = 'cail_0518/data_test_seg.txt'
out_file_test_name = 'cail_0518/data_test_fasttext_num.txt'

def write_file(x_list,y_list,out_file):
    for i in range(len(x_list)):
        out_str = ''
        for y in y_list[i]:
            out_str = out_str + '__label__' + y + ' '
        out_str += ' '.join([str(a) for a in x_list[i]])
        out_file.write(out_str + '\n')

with open(('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE), 'rb') as f:
        dataset = pickle.load(f)
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

out_file_train = open(out_file_train_name,'w')
out_file_test = open(out_file_test_name,'w')
write_file(train_x,train_y,out_file_train)
write_file(test_x,test_y,out_file_test)


# f = open(file_name,'r')
# f_out = open(out_file_name,'w')
# for line in f:
#     out_str = ''
#     line_list = line.strip().split('\t')
#     sentence = line_list[0]
#     label_str = line_list[1]
#     label_list = label_str.split('&')
#     for label in label_list:
#         out_str = out_str + '__label__' + label + ' '
#     out_str += sentence
#     f_out.write(out_str + '\n')

