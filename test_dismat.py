import numpy as np
MAX_VOCAB_SIZE = 60702
import pickle
dist_mat_list = np.load('aux_files/sdist_mat_dic_%d.npy' % (MAX_VOCAB_SIZE))
dist_mat_order = np.load('aux_files/sdist_order_%d.npy' % (MAX_VOCAB_SIZE))
with open('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)

# print(np.shape(dist_mat_list))
# print(dist_mat_list[200:205])

# print(np.shape(dist_mat_order))
# print(dist_mat_order[200:205])


# for i in range(60700,60703):
#     cnt_i = i
#     if i == 0:
#         cnt_i = MAX_VOCAB_SIZE
#     print(dataset.inv_dict[cnt_i])
#     for j in range(101):
#         cnt_dist = dist_mat_order[cnt_i][j]
#         if dist_mat_order[cnt_i][j] == 0:
#             cnt_dist = MAX_VOCAB_SIZE
#         print(cnt_dist, dataset.inv_dict[cnt_dist], dist_mat_list[cnt_i][j])


def pick_most_similar_words(src_word, dist_mat_list, dist_mat_order, ret_count=10, threshold=None):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    # dist_order = np.argsort(dist_mat[src_word,:])[1:1+ret_count]
    # dist_list = dist_mat[src_word][dist_order]
    dist_order = dist_mat_order[src_word][1:ret_count+1]
    dist_list = dist_mat_list[src_word][1:ret_count+1]
    # print(dist_order)
    # print(dist_list)
    if dist_list[-1] == 0:
        return [], []
    mask = np.ones_like(dist_list)
    if threshold is not None:
        mask = np.where(dist_list < threshold)
        return dist_order[mask], dist_list[mask]
    else:
        return dist_order, dist_list


print(pick_most_similar_words(100, dist_mat_list, dist_mat_order))
print(pick_most_similar_words(100, dist_mat_list, dist_mat_order, threshold=1.45))