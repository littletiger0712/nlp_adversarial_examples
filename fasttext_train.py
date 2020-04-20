import fasttext
import pickle
import numpy as np
max_len = 250
MAX_VOCAB_SIZE = 50000

# from keras.preprocessing.sequence import pad_sequences
# with open(('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE), 'rb') as f:
#         dataset = pickle.load(f)
# train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
# train_y = np.array(dataset.train_y)

# print(len(train_x))
# print(train_x[0])
# print(len(train_y))
# print(train_y[0])
# model = fasttext.train_supervised(input="cail_0518/data_train_fasttext_num.txt")
# # model = fasttext.load_model("cail_0518/fasttext_model.bin")
# print(model.test('cail_0518/data_test_fasttext_num.txt'))
# print(model.predict(['16205 28415 59178 1403 24193 50131 3314 37462 6109 47819 58669 39413 18053 30370 46951 32242 9379 58162 57301 60090 13542 48641 39776 11752 30370 3398 54928 25104 40561 12299 46458 58162 33464 52480 38295 55139 1403 3314 25607 25608 26933 37462 55430 37553 58162 32159 17167 57459 23487 58162 19024 9023 13167 19024 23183 18053 43640 10924 1814 15023 1403 11005 25287 49733 1403 56170 55171 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0']))
# model.save_model("cail_0518/fasttext_model.bin")


model = fasttext.load_model("cail_0518/fasttext_model.bin")
print(model.test('cail_0518/data_test_fasttext_num.txt'))
print(model.predict(['16205 28415 59178 1403 24193 50131 3314 37462 6109 47819 58669 39413 18053 30370 46951 32242 9379 58162 57301 60090 13542 48641 39776 11752 30370 3398 54928 25104 40561 12299 46458 58162 33464 52480 38295 55139 1403 3314 25607 25608 26933 37462 55430 37553 58162 32159 17167 57459 23487 58162 19024 9023 13167 19024 23183 18053 43640 10924 1814 15023 1403 11005 25287 49733 1403 56170 55171 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'],k=20))