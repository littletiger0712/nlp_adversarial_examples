import numpy as np
from progress.bar import Bar



# MAX_VOCAB_SIZE = 500
MAX_VOCAB_SIZE = 60702
STORE_SIZE = 100
embedding_matrix = np.load(('aux_files/embeddings_counter_%d.npy' % (MAX_VOCAB_SIZE))) # c*n
# missed = np.load(('aux_files/missed_embeddings_counter_%d.npy' % (MAX_VOCAB_SIZE)))
# c_ = -2*np.dot(embedding_matrix.T, embedding_matrix) # n*n
# a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1))
# b = a.T
# dist_mat = a+b+c_ # n*n
# print('distence matrix build success!')
# dist_order = np.argsort(dist_mat, axis=1)[:,:STORE_SIZE+1]
# idx = (np.arange(MAX_VOCAB_SIZE+1) * (MAX_VOCAB_SIZE+1)).reshape(-1, 1)
# idx = dist_order + idx
# dist_mat_dic = dist_mat.flatten()[idx].reshape(MAX_VOCAB_SIZE+1, STORE_SIZE+1)


dist_mat_dic = np.zeros((MAX_VOCAB_SIZE+1, STORE_SIZE+1))
dist_order = np.zeros((MAX_VOCAB_SIZE+1, STORE_SIZE+1), dtype=np.int)
bar = Bar('test', max=MAX_VOCAB_SIZE+1)
for i in range(MAX_VOCAB_SIZE+1):
    item_embedding = embedding_matrix[:, i].reshape(-1, 1)
    distance_vec = np.linalg.norm(embedding_matrix-item_embedding, ord=2, axis=0)
    dist_order[i] = np.argsort(distance_vec)[:STORE_SIZE+1]
    dist_mat_dic[i] = distance_vec[dist_order[i]]
    bar.next()

np.save(('aux_files/sdist_mat_dic_%d.npy' % (MAX_VOCAB_SIZE)), dist_mat_dic)
np.save(('aux_files/sdist_order_%d.npy' % (MAX_VOCAB_SIZE)), dist_order)
