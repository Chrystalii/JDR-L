import logging
import os
import random

import numpy as np
from keras.layers import Embedding, dot
from keras.layers import Input, Dense, Activation
from keras.layers import Merge, Reshape
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_logger():
    logger = logging.getLogger('item2vec')
    formatter = logging.Formatter('[%(asctime)s] %(levelname).1s: %(message)s')
    # StreamHandler for print log to console
    hdr = logging.StreamHandler()
    hdr.setFormatter(formatter)
    hdr.setLevel(logging.DEBUG)
    logger.addHandler(hdr)
    logger.setLevel(logging.INFO)
    return logger


def skipgram_model(vocab_size, embedding_dim=100, paradigm='Functional'):
    # Sequential paradigm
    if paradigm == 'Sequential':
        target = Sequential()
        target.add(Embedding(vocab_size, embedding_dim, input_length=1))
        context = Sequential()
        context.add(Embedding(vocab_size, embedding_dim, input_length=1))

        # merge the pivot and context models
        model = Sequential()
        model.add(Merge([target, context], mode='dot'))
        model.add(Reshape((1,), input_shape=(1, 1)))
        model.add(Activation('sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    # Functional paradigm
    elif paradigm == 'Functional':
        target = Input(shape=(1,), name='target')
        context = Input(shape=(1,), name='context')
        # print target.shape, context.shape
        shared_embedding = Embedding(vocab_size, embedding_dim, input_length=1, name='shared_embedding')
        embedding_target = shared_embedding(target)
        embedding_context = shared_embedding(context)
        # print embedding_target.shape, embedding_context.shape

        merged_vector = dot([embedding_target, embedding_context], axes=-1)
        reshaped_vector = Reshape((1,), input_shape=(1, 1))(merged_vector)
        # print merged_vector.shape
        prediction = Dense(1, input_shape=(1,), activation='sigmoid')(reshaped_vector)
        # print prediction.shape

        model = Model(inputs=[target, context], outputs=prediction)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    else:
        print('paradigm error')
        return None


def skipgram_reader_generator(sequences, num_items, context_window=2):
    def reader():
        for seq in sequences:
            for i in range(len(seq)):
                target = seq[i]
                # generate positive sample
                context_list = []
                j = i - context_window
                while j <= i + context_window and j < len(seq):
                    if j >= 0 and j != i:
                        context_list.append(seq[j])
                        yield ((target, seq[j]), 1)
                    j += 1
                # generate negative sample
                for _ in range(len(context_list)):
                    ne_idx = random.randrange(0, num_items)
                    while ne_idx in context_list:
                        ne_idx = random.randrange(0, num_items)
                    yield ((target, ne_idx), 0)

    return reader


def shuffle(reader, buf_size):
    def data_reader():
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []
        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader


def normalize_item_ids(sequences):
    item_counter = {}
    for seq in sequences:
        for item in seq:
            if item not in item_counter:
                item_counter[item] = 0
            item_counter[item] += 1
    item_counter_sorted = sorted(item_counter.items(), key=lambda x: (-x[1], x[0]))
    items, _ = list(zip(*item_counter_sorted))
    item_dict = dict(zip(items, range(len(items))))
    new_seqs = []
    for seq in sequences:
        new_seq = []
        for item in seq:
            new_seq.append(item_dict[item])
        new_seqs.append(new_seq)
    return item_dict, new_seqs


def item2vec(source_sequences, emb_size=64, epochs=50, context_window_size=2, batch_size=256, is_normalize=False, return_sequences=False):
    # network conf
    logger = get_logger()
    logger.info('start item2vec')
    paradigm = 'Functional'
    item_dict, sequences = normalize_item_ids(source_sequences)
    num_items = len(item_dict)
    model = skipgram_model(num_items, emb_size, paradigm)
    logger.info('start train item2vec %s' % str(model.layers))
    for epoch_id in range(epochs):
        # train by batch
        batch_id = 0
        x_batch = [[], []]
        y_batch = []
        loss_list = []
        for movie_ids, label in shuffle(skipgram_reader_generator(sequences, num_items, context_window=context_window_size), 10000)():
            batch_id += 1
            x_batch[0].append(movie_ids[0])
            x_batch[1].append(movie_ids[1])
            y_batch.append(label)
            if batch_id % (batch_size * 1000) == 0:
                # Print evaluate log
                logger.info('[epoch #%d] batch #%d, train loss:%s' % (epoch_id, batch_id, np.mean(loss_list)))
                loss_list = []
            if batch_id % batch_size == 0:
                x = [np.array(x_batch[0]), np.array(x_batch[1])]
                loss = model.train_on_batch(x, np.array(y_batch))
                loss_list.append(loss)
                x_batch = [[], []]
                y_batch = []
    logger.info('item2vec train done')

    item_idx_dict = dict([(v, k) for k, v in item_dict.items()])
    item_vecs = {}
    for idx, vec in enumerate(model.layers[2].get_weights()[0].tolist()):
        item_vecs[item_idx_dict[idx]] = np.array(vec) / np.sqrt(np.dot(vec, vec)) if is_normalize else np.array(vec)

    if return_sequences:
        outputs = []
        for seq in source_sequences:
            output = []
            for item in seq:
                output.append(item_vecs[item])
            outputs.append(output)
        logger.info('create item2vec outputs done, items = %d, seqs = %d' % (len(item_vecs), len(outputs)))
        return item_vecs, np.array(outputs)
    else:
        logger.info('create item2vec outputs done, items = %d' % len(item_vecs))
        return item_vecs


def fast_item2vec(source_sequences, emb_size=64, epochs=50, context_window_size=2, batch_size=256, is_normalize=False, return_sequences=False):
    # network conf
    logger = get_logger()
    logger.info('start item2vec')
    paradigm = 'Functional'
    item_dict, sequences = normalize_item_ids(source_sequences)
    num_items = len(item_dict)
    model = skipgram_model(num_items, emb_size, paradigm)
    logger.info('start train item2vec %s' % str(model.layers))

    batch_id = 0
    x_train = [[], []]
    y_train = []
    x_batch = [[], []]
    y_batch = []
    for movie_ids, label in shuffle(skipgram_reader_generator(sequences, num_items, context_window=context_window_size), 10000)():
        batch_id += 1
        x_batch[0].append(movie_ids[0])
        x_batch[1].append(movie_ids[1])
        y_batch.append(label)
        if batch_id % batch_size == 0:
            x_train[0].extend(x_batch[0])
            x_train[1].extend(x_batch[1])
            y_train.extend(y_batch)
            x_batch = [[], []]
            y_batch = []

    x_train[0] = np.array(x_train[0])
    x_train[1] = np.array(x_train[1])
    y_train = np.array(y_train)
    print('x_train:', x_train[0].shape, x_train[1].shape)
    print('y_train:', y_train.shape)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    logger.info('item2vec train done')

    item_idx_dict = dict([(v, k) for k, v in item_dict.items()])
    item_vecs = {}
    for idx, vec in enumerate(model.layers[2].get_weights()[0].tolist()):
        item_vecs[item_idx_dict[idx]] = np.array(vec) / np.sqrt(np.dot(vec, vec)) if is_normalize else np.array(vec)

    if return_sequences:
        outputs = []
        for seq in source_sequences:
            output = []
            for item in seq:
                output.append(item_vecs[item])
            outputs.append(output)
        logger.info('create item2vec outputs done, items = %d, seqs = %d' % (len(item_vecs), len(outputs)))
        return item_vecs, np.array(outputs)
    else:
        logger.info('create item2vec outputs done, items = %d' % len(item_vecs))
        return item_vecs


def create_item_vecs(item_seqs, emb_size, epochs, output_file, batch_size=256, is_fast=False):
    # user_item_ts_list = np.concatenate([dataset.user_ids.reshape((-1, 1)), dataset.item_ids.reshape((-1, 1)), dataset.timestamps.reshape((-1, 1))], 1)
    # user_item_seqs = {}
    # for x in user_item_ts_list:
    #     uid = x[0]
    #     iid = [x[1], x[2]]
    #     if uid not in user_item_seqs:
    #         user_item_seqs[uid] = []
    #     user_item_seqs[uid].append(iid)
    #
    # item_seqs = []
    # for iids in item_sequences:
    #     sorted_iids = list(sorted(iids, key=lambda x: x[1]))
    #     item_seqs.append(np.array([x[0] for x in sorted_iids]))
    if is_fast:
        item_dict = fast_item2vec(item_seqs, emb_size=emb_size, epochs=epochs, batch_size=batch_size)
    else:
        item_dict = item2vec(item_seqs, emb_size=emb_size, epochs=epochs, batch_size=batch_size)
    with open(output_file, 'w') as f:
        for k, v in item_dict.items():
            f.write('%d %s\n' % (k, ' '.join([str(_) for _ in v])))


def get_item_vecs(vec_file):
    lines = np.loadtxt(vec_file)
    outputs = {}
    max_len = 0.0
    for line in lines:
        vec = np.array(line[1:], dtype=np.float)
        vec = vec / np.sqrt(np.dot(vec, vec))
        # max_len = max(max_len, np.sqrt(np.dot(vec, vec)))
        outputs[int(line[0])] = vec

    # for k in outputs.keys():
    #     outputs[k] = outputs[k] / max_len
    return outputs
