import math
import os

import keras.backend as K
import numpy as np
from keras.layers import Input, LSTM, Permute, Lambda, RepeatVector, Flatten, TimeDistributed, merge, Dense, Activation, Dropout
from keras.models import Model

from dataset import load_dataset, get_item_seqs, get_train, get_test, save_2d_array
from item2vec import create_item_vecs, get_item_vecs
import cupy
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.2"

data_path = '../data/amazon_movies_tv.rating'
output_dir = './output_movie_dropout'
# data_path = '../data/ml100k.rating'
# output_dir = './output_100k_dropout'

item_vec_file = os.path.join(output_dir, 'item_vecs')

# lstm hidden size for long-term
n_lstm1_hidden = np.random.randint(12,500)
# lstm hidden size for short-term
n_lstm2_hidden = n_lstm1_hidden

# epochs for item2vec
n_item2vec_epochs = np.random.randint(2, 50)

# epochs for mpr
n_epochs = np.random.randint(10, 300)
# batch size for mpr
batch_size = 2048

# train dataset size (count by sequence)
n_train_slice = 0

# sequence length
n_steps = np.sample([10,20,40,60,80], 1)
# union-level length
n_short_steps = np.sample([2,4,6,8], 1)

# walk size between sequences
step_size = n_short_steps

# embeding size
n_features = n_lstm1_hidden

# test / dataset,default=0.2
split_percent = 0.2


def embed_data(sequences, item_vecs, embed_size):
    outputs = []
    for seq in sequences:
        output = []
        for item in seq:
            if item == 0:
                output.append(np.zeros(embed_size))
            else:
                output.append(item_vecs[item])
        outputs.append(np.array(output))
    outputs = np.array(outputs)
    return outputs[:, :-1, :], outputs[:, -1, :]


def predict(model, item_vecs, test, test_uids, train_user_items):
    if len(test) > 10000:
        sample_indices = (np.random.sample(int(len(test) / 10.0)) * len(test)).astype(np.int)
        test = np.array(test)[sample_indices]
        test_uids = np.array(test_uids)[sample_indices]

    x_test, y_test = embed_data(test, item_vecs, n_features)
    print('x_test', x_test.shape, 'y_test', y_test.shape)
    y_pred = model.predict(x_test)
    save_2d_array(output_dir, 'y_pred', y_pred)

    def calc_rank(use_all):
        found_count = 0
        mrr = []
        ndcd = []
        total_idx = []
        all_iids = range(len(item_vecs))
        total_test = len(test)
        start = time.time()
        for i in range(total_test):
            if i % 1000 == 0:
                print('predict: %d of %d, duration = %ds. found %d' % (i, total_test, time.time() - start, found_count))
                start = time.time()
            y_t = test[i, -1]
            y_p = y_pred[i]
            if use_all:
                candidate_iids = item_vecs.items()
            else:
                train_iids = train_user_items[test_uids[i]]
                candidates = np.setdiff1d(all_iids, train_iids)
                candidate_iids = [(k, item_vecs[k]) for k in candidates]
            cks = [x for x, _ in candidate_iids]
            true_idx = cks.index(y_t)
            if true_idx == -1:
                total_idx.append(len(candidate_iids))
                mrr.append(0.0)
                ndcd.append(0.0)
                continue

            cmat = cupy.array([v for _, v in candidate_iids], dtype=cupy.float32)
            cpred = cupy.array(y_p, dtype=cupy.float32)
            cdot = cupy.matmul(cmat, cpred)
            indices = cupy.argsort(-cdot)
            rank = int(cupy.where(indices == true_idx)[0][0])

            total_idx.append(rank)
            # print fc
            if rank < 20:
                found_count += 1
                mrr.append(1.0 / (rank + 1))
                ndcd.append(1.0 / math.log(rank + 2, 2))
            else:
                mrr.append(0.0)
                ndcd.append(0.0)
        print('use_all:', use_all)
        print('found:', found_count, ', mrr:', np.mean(mrr), 'ndcg:', np.mean(ndcd))
        print('avg_index:', np.mean(total_idx))

    calc_rank(True)
    # calc_rank(False)


def train_model(x_train, y_train):
    x_input = Input(shape=(n_steps, n_features))
    activations = LSTM(n_lstm1_hidden, input_shape=(n_steps, n_features), return_sequences=True)(x_input)
    activations_n = Lambda(lambda x, n: x[:, -n:, :], arguments={'n': n_short_steps})(activations)

    # compute importance for each step
    attention = TimeDistributed(Dense(1))(activations_n)
    attention = Flatten()(attention)
    attention = Activation('sigmoid')(attention)
    attention = RepeatVector(n_lstm1_hidden)(attention)
    # alpha
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = merge([activations_n, attention], mode='mul')
    s1 = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    s2 = LSTM(n_lstm2_hidden, input_shape=(n_steps, n_lstm1_hidden))(sent_representation)

    # s1 = Dense(n_features, activation='softmax')(s1)
    # s2 = Dense(n_features, activation='softmax')(s2)
    combo = merge([s1, s2], mode='sum')
    combo = Dense(n_features, activation='sigmoid')(combo)

    model = Model(inputs=x_input, outputs=combo)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model


def train_softmax(x_train, y_train):
    x_input = Input(shape=(n_steps, n_features))
    activations = LSTM(n_lstm1_hidden, activation='relu', input_shape=(n_steps, n_features), return_sequences=True)(x_input)
    activations_n = Lambda(lambda x, n: x[:, -n:, :], arguments={'n': n_short_steps})(activations)

    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations_n)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(n_lstm1_hidden)(attention)
    # alpha
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = merge([activations_n, attention], mode='mul')
    s1 = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    s2 = LSTM(n_lstm2_hidden, input_shape=(n_steps, n_lstm1_hidden))(sent_representation)

    s1 = Dense(n_features, activation='softmax')(s1)
    s2 = Dense(n_features, activation='softmax')(s2)
    combo = merge([s1, s2], mode='sum')
    # combo = Dense(n_features, activation='softmax')(combo)

    model = Model(inputs=x_input, outputs=combo)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model

def lstm_only(x_train, y_train):
    x_input = Input(shape=(n_steps, n_features))
    activations = LSTM(n_lstm1_hidden, activation='relu', input_shape=(n_steps, n_features))(
        x_input)
    output = Dense(n_features, activation='softmax')(activations)
    model = Model(inputs=x_input, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model

def lstm_individual(x_train, y_train):
    x_input = Input(shape=(n_steps, n_features))
    activations = LSTM(n_lstm1_hidden, activation='relu', input_shape=(n_steps, n_features), return_sequences=True)(
        x_input)
    activations_n = Lambda(lambda x, n: x[:, -n:, :], arguments={'n': n_short_steps})(activations)
    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations_n)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(n_lstm1_hidden)(attention)
    # alpha
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = merge([activations_n, attention], mode='mul')
    s1 = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
    s1 = Dense(n_features, activation='softmax')(s1)

    model = Model(inputs=x_input, outputs=s1)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model

def lstm_union(x_train, y_train):
    x_input = Input(shape=(n_steps, n_features))
    activations = LSTM(n_lstm1_hidden, activation='relu', input_shape=(n_steps, n_features), return_sequences=True)(
        x_input)
    activations_n = Lambda(lambda x, n: x[:, -n:, :], arguments={'n': n_short_steps})(activations)

    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations_n)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(n_lstm1_hidden)(attention)
    # alpha
    attention = Permute([2, 1])(attention)

    # apply the attention
    sent_representation = merge([activations_n, attention], mode='mul')
    s2 = LSTM(n_lstm2_hidden, input_shape=(n_steps, n_lstm1_hidden))(sent_representation)
    s2 = Dense(n_features, activation='softmax')(s2)

    model = Model(inputs=x_input, outputs=s2)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model

def train_softmax_iu(x_train, y_train):
    x_input = Input(shape=(n_steps, n_features))

    # let n_steps = 10, n_short_steps = 2
    # x1, x2, ..., x10
    activations = LSTM(n_lstm1_hidden, activation='relu', input_shape=(n_steps, n_features), return_sequences=True)(x_input)

    # x9, x10
    activations_n = Lambda(lambda x, n: x[:, -n:, :], arguments={'n': n_short_steps})(activations)

    # compute importance for each step
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations_n)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(n_lstm1_hidden)(attention)

    # alpha
    # a9, a10
    attention = Permute([2, 1])(attention)

    # apply the attention
    # a9 * h9, a10 * h10
    sent_representation = merge([activations, activations_n], mode='mul')
    # a9 * h9 + a10 * h10
    s1 = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    lstm2_input = Lambda(lambda x, n, steps, lstm1: K.concatenate([K.reshape(K.sum(x[:, i:i + n, :], axis=1), (-1, 1, lstm1)) for i in range(steps - n)], axis=1),
                         arguments={'steps': n_steps, 'n': n_short_steps, 'lstm1': n_lstm1_hidden})(activations)
    # h11
    s2 = LSTM(n_lstm2_hidden, input_shape=(n_steps - n_short_steps, n_lstm1_hidden))(lstm2_input)

    s1 = Dense(n_features, activation='softmax')(s1)
    s2 = Dense(n_features, activation='softmax')(s2)
    combo = merge([s1, s2], mode='sum')
    # combo = Dense(n_features, activation='softmax')(combo)

    model = Model(inputs=x_input, outputs=combo)
    model.compile(Adam(lr=0.001), loss='mse', metrics=['accuracy'])
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, verbose=1)
    return model

def prepare_dataset():
    np.random.seed(7)
    load_dataset(data_path, output_dir, split_percent=split_percent, seq_len=n_steps + 1, seq_steps=step_size, is_leave_one_out=False)

    item_seqs = get_item_seqs(output_dir)
    create_item_vecs(item_seqs, n_features, n_item2vec_epochs, item_vec_file, is_fast=False, batch_size=256)


def run():
    np.random.seed(1337)
    # np.random.seed(7)
    train = get_train(output_dir)
    np.random.shuffle(train)
    item_vecs = get_item_vecs(item_vec_file)

    if n_train_slice > 0:
        x_train, y_train = embed_data(train[:n_train_slice], item_vecs, n_features)
    else:
        x_train, y_train = embed_data(train, item_vecs, n_features)
    print('x_train', x_train.shape, 'y_train', y_train.shape)

    model = train_softmax(x_train, y_train)

    print('--------- predict test --------')
    test, test_uids, train_user_items = get_test(output_dir)
    print(test.shape)
    print(test_uids.shape)
    print(len(train_user_items))
    if n_train_slice > 0:
        test_slice = int(n_train_slice * split_percent / (1 - split_percent))
        test = test[-test_slice:]
        test_uids = test_uids[-test_slice:]
    predict(model, item_vecs, test, test_uids, train_user_items)


if __name__ == '__main__':
    prepare_dataset()
    run()
