import math
import os

import numpy as np
import pandas as pd


def split_item_to_sequence(items, seq_len=20, seq_steps=10):
    seqs = []
    item_len = len(items)
    if item_len < seq_len:
        seqs.append(np.pad(items, (seq_len - item_len, 0), 'constant'))
    else:
        start = 0
        while start + seq_len < item_len:
            seqs.append(items[start: start + seq_len])
            start += seq_steps

        if start + 1 < item_len:
            seqs.append(items[-seq_len:])
    return seqs


def load_dataset(datapath, output_dir, userthreshold=10, split_percent=0.2, seq_len=20, seq_steps=10, is_leave_one_out=False):
    """
        filter out user and items whoes number of neighbors is less threshold
        save userSequeces, trainSequences, testSequences, testUids, trainUserItems
    """
    data = pd.read_csv(datapath, sep="\t", header=None, usecols=[0, 1, 3], dtype=np.int32)
    data.columns = ['user', 'item', 'timestamp']
    print('origin data:', data.shape)

    users = data.groupby("user").size()
    filtered_users = np.in1d(data.user, users[users > userthreshold].index)
    data = data[filtered_users]
    users = np.unique(data.user)
    user_len = users.shape[0]
    print('filtered data:', data.shape, 'users:', users.shape)

    # drop_out_percent = 0.2
    # users = users[np.unique((np.random.sample(int(user_len * drop_out_percent)) * user_len).astype(np.int))]
    # data = data[np.in1d(data.user, users)]
    # data = data.sample(frac=drop_out_percent)
    print('dropout data:', data.shape, 'users:', users.shape)

    data = data.sort_values(by=['user', 'timestamp'], ascending=True)

    users = dict([(v, index) for index, v in enumerate(data.user.unique())])
    items = dict([(v, index) for index, v in enumerate(data.item.unique())])
    print('users:', len(users))
    print('items:', len(items))

    uid_iids = dict()
    for u, i, _ in data.values:
        uid = users[u]
        iid = items[i]
        if uid not in uid_iids:
            uid_iids[uid] = []
        uid_iids[uid].append(iid)

    train_seqs = []
    test_seqs = []
    train_user_items = {}
    test_uids = []
    for u, iids in uid_iids.items():
        if is_leave_one_out:
            train_its = iids[:-1]
            test_its = iids[-seq_len:]
        else:
            test_size = int(math.ceil(len(iids) * split_percent))
            train_its = iids[:-test_size]
            test_its = iids[-test_size:]

        if len(train_its) == 0 or len(test_its) == 0:
            continue

        train_seqs.extend(split_item_to_sequence(train_its, seq_len, seq_steps))
        test_subseqs = split_item_to_sequence(test_its, seq_len, seq_steps)
        test_seqs.extend(test_subseqs)
        train_user_items[u] = np.unique(train_its)
        # candidate_iids = np.setdiff1d(all_iids, train_its)
        # candidate_iids = np.unique(np.concatenate([candidate_iids, np.unique(test_its)]))
        for _ in range(len(test_subseqs)):
            test_uids.append(u)

    uid_iid_seqs = np.array(list(uid_iids.values()))
    train_seqs = np.array(train_seqs, dtype=np.int)
    test_seqs = np.array(test_seqs, dtype=np.int)
    test_uids = np.array(test_uids, dtype=np.int)
    print('uid_iid_seqs:', uid_iid_seqs.shape)
    print('train_seqs:', train_seqs.shape)
    print('test_seqs:', test_seqs.shape)
    print('test_uids:', test_uids.shape)
    print('train_iids:', len(train_user_items))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_2d_array(output_dir, 'item_seqs', uid_iid_seqs)
    save_2d_array(output_dir, 'train', train_seqs)
    save_2d_array(output_dir, 'test', test_seqs)
    save_2d_array(output_dir, 'test_uids', test_uids.reshape(test_uids.shape[0], 1))
    save_train_user_items(output_dir, train_user_items)


def save_2d_array(output_dir, file_name, arr):
    with open(os.path.join(output_dir, file_name), 'w') as f:
        for l in arr:
            f.write('%s\n' % ' '.join([str(_) for _ in l]))


def load_2d_array(output_dir, file_name, converter=lambda x: int(x)):
    arr = []
    with open(os.path.join(output_dir, file_name), 'r') as f:
        for l in f.readlines():
            arr.append([converter(x) for x in l.strip('\n').split(' ')])
    return np.array(arr)


def save_train_user_items(output_dir, train_user_items):
    with open(os.path.join(output_dir, 'train_user_items'), 'w') as f:
        for k, l in train_user_items.items():
            f.write('%d %s\n' % (k, ' '.join([str(_) for _ in l])))


def load_train_user_items(output_dir):
    train_user_items = {}
    with open(os.path.join(output_dir, 'train_user_items'), 'r') as f:
        for l in f.readlines():
            ids = [int(x) for x in l.strip('\n').split(' ')]
            train_user_items[ids[0]] = ids[1:]
    return train_user_items


def get_item_seqs(output_dir):
    return load_2d_array(output_dir, 'item_seqs')


def get_train(output_dir):
    return load_2d_array(output_dir, 'train')


def get_test(output_dir):
    return load_2d_array(output_dir, 'test'), load_2d_array(output_dir, 'test_uids').reshape(-1), load_train_user_items(output_dir)


if __name__ == '__main__':
    # print(split_item_to_sequence([1, 2, 3, 4, 5], 5))
    # print(int(0.0))
    # s = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 365\n'
    # print([int(x) for x in s.strip('\n').split(' ')])
    dir_path = './output_movie_dropout'
    print(load_train_user_items(dir_path))
    # seqs, train, test, candidate = get_dataset('../data/ml100k.rating')
    # print seqs.shape
    # print train.shape
    # print test.shape
    # print candidate.shape
    #
    # print '--------'
    # print seqs[0]
    # print train[0:4]
    # print test[0]
    # print candidate[0]
