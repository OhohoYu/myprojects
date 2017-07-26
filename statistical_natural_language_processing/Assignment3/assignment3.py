%%capture
%load_ext autoreload
%autoreload 2
%matplotlib inline
#! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY (RIEDEL)
import sys, os
_snlp_book_dir = "../../../../../"
sys.path.append(_snlp_book_dir)
# docker image contains tensorflow 0.10.0rc0. We will support execution of only that version!
import statnlpbook.nn as nn

import tensorflow as tf
import numpy as np

#! SETUP 2 - DO NOT CHANGE, MOVE NOR COPY
data_path = _snlp_book_dir + "data/nn/"
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
assert(len(data_train) == 45502)

### PRE-PROCESSING PIPELINE

### OUR PIPELINE
import re

### REGEXP TOKENISER
def tokenize(input):
    token = re.compile('\w+|[.?:,!()]')
    return token.findall(input)

## PIPELINE IS MODIFIED TO ACCOUNT FOR TOKENISATION
def pipeline(data, vocab=None, max_sent_len_=None):
    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1}

    max_sent_len = -1
    data_sentences = []
    data_sentences_len = []
    data_orders = []
    for instance in data:
        sents = []
        sents_len = []
        for sentence in instance['story']:
            sent = []
            tokenized = tokenize(sentence)
            for token in tokenized:
                if not is_ext_vocab and token not in vocab:
                    vocab[token] = len(vocab)
                if token not in vocab:
                    token_id = vocab['<OOV>']
                else:
                    token_id = vocab[token]
                sent.append(token_id)
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            sents.append(sent)
            sents_len.append(len(sent))
        data_sentences.append(sents)
        data_sentences_len.append(sents_len)
        data_orders.append(instance['order'])

    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_
    out_sentences = np.full([len(data_sentences), 5, max_sent_len],
                            vocab['<PAD>'], dtype=np.int32)

    for i, elem in enumerate(data_sentences):
        for j, sent in enumerate(elem):
            out_sentences[i, j, 0:len(sent)] = sent

    out_sentences_len = np.array(data_sentences_len, dtype=np.int32)
    out_orders = np.array(data_orders, dtype=np.int32)

    return out_sentences, out_sentences_len, out_orders, vocab

# convert train set to integer IDs
train_stories, train_stories_len, train_orders, vocab = pipeline(data_train)

# get the length of the longest sentence
max_sent_len = train_stories.shape[2]

# convert dev set to integer IDs, based on the train vocabulary and max_sent_len
dev_stories, dev_stories_len, dev_orders, _ = pipeline(data_dev, vocab=vocab,
                                                       max_sent_len_=max_sent_len)

### pre-trained word embeddings - new words loaded as OOV (see pipeline)
def load_glove(path):
    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def filter_embeddings(embeddings, vocab):
    new_embeddings = {}
    for word in vocab:
        if word in embeddings:
            new_embeddings[word] = embeddings[word]
        elif word.lower() in embeddings:
            new_embeddings[word.lower()] = embeddings[word.lower()]
    return new_embeddings

# embeddings_dict = load_glove('glove.6B/glove.6B.200d.txt')
# embeddings_dict = filter_embeddings(embeddings_dict, vocab.keys())
# np.save('embeddings_dict_200d.npy', embeddings_dict)

# embeddings_dict = np.load('embeddings_dict_200d.npy').item()
# print('embeddings loaded')

'''
LAYER NORMALISATION (IN FINAL MODEL)
In our final model, we make use of the function LayerNormalizedLSTMCell. This function has been adapted from
a r2t2.com implementation (see link below).   Layer normalisation is a feature
recently published by Lei Ba et al. (2016). The function LayerNormalizedLSTMCell is essentially an edit of
the built-in Tensorflow function 'tf.nn.rnn_cell.LSTMCell'. It accounts for layer normalisation applying
the subfunction 'ln' to each gate output in a LSTM cell. The 'ln' function implements the layer normalisation,
normalising to a variance of 1 and a mean of 0 a linear transformation output. Layer normalisation improves
the performance of our model. More details on the following link:

http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html
'''
def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift

class LayerNormalizedLSTMCell(tf.nn.rnn_cell.RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # change bias argument to False since LN will add bias via shift
            concat = tf.nn.rnn_cell._linear([inputs, h], 4 * self._num_units, False)

            i, j, f, o = tf.split(1, 4, concat)

            # add layer normalization to each gate
            i = ln(i, scope = 'i/')
            j = ln(j, scope = 'j/')
            f = ln(f, scope = 'f/')
            o = ln(o, scope = 'o/')

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                   self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope = 'new_h/')) * tf.nn.sigmoid(o)
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state

### MODEL PARAMETERS ###
target_size = 5
vocab_size = len(vocab)
n_rnn_layers = 2
input_size = 200
lstm_size = 50
fc_size = 200
output_size = 5
dropout = 0.9
learning_rate = 0.01

'''
Our final model makes use of truncated_normal technique for embedding initialisation.
Random values are sampled from a normal distribution.
Values with a magnitude greater than 2 standard deviations from the mean are not used and are resampled.
'''
def truncated_normal(shape, mean, std):
    with tf.Session() as sess:
        x = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=std,
                                            dtype=np.float32))
        sess.run(tf.initialize_all_variables())
        return x.eval()

### EMBEDDING CREATION
def create_embeddings_W(embeddings_dict, vocab):
    # compute statistics of pretrained embeddings
    E = np.stack(list(embeddings_dict.values()))
    mean = np.mean(E, axis=0)
    std = np.std(E, axis=0)

    rand_count = 0
    vocab_size = len(vocab)
    # embeddings_W = np.zeros((vocab_size, input_size), dtype=np.float32)
    # embedding = np.random.uniform(-0.1, 0.1, input_size)
    embeddings_W = truncated_normal([vocab_size, input_size], np.mean(mean), np.mean(std))
    for word, index in vocab.items():
        if word in embeddings_dict:
            embedding = embeddings_dict[word]
            embeddings_W[index,:] = embedding
        elif word.lower() in embeddings_dict:
            embedding = embeddings_dict[word.lower()]
            embeddings_W[index,:] = embedding
        else:
            rand_count += 1

    print('embedding W created winth %d rnad inits'%rand_count)
    return embeddings_W

# embeddings_W = create_embeddings_W(embeddings_dict, vocab)
# np.save('embeddings_W.npy', embeddings_W)

# embeddings_W = np.load('embeddings_W_200d.npy')
# print('embeddings_W loaded')

'''
Our final model makes use of truncated_normal technique for embedding initialisation.
Random values are sampled from a normal distribution.
Values with a magnitude greater than 2 standard deviations from the mean are not used and are resampled.
'''
def truncated_normal(shape, mean, std):
    with tf.Session() as sess:
        x = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=std,
                                            dtype=np.float32))
        sess.run(tf.initialize_all_variables())
        return x.eval()

### EMBEDDING CREATION
def create_embeddings_W(embeddings_dict, vocab):
    # compute statistics of pretrained embeddings
    E = np.stack(list(embeddings_dict.values()))
    mean = np.mean(E, axis=0)
    std = np.std(E, axis=0)

    rand_count = 0
    vocab_size = len(vocab)
    # embeddings_W = np.zeros((vocab_size, input_size), dtype=np.float32)
    # embedding = np.random.uniform(-0.1, 0.1, input_size)
    embeddings_W = truncated_normal([vocab_size, input_size], np.mean(mean), np.mean(std))
    for word, index in vocab.items():
        if word in embeddings_dict:
            embedding = embeddings_dict[word]
            embeddings_W[index,:] = embedding
        elif word.lower() in embeddings_dict:
            embedding = embeddings_dict[word.lower()]
            embeddings_W[index,:] = embedding
        else:
            rand_count += 1

    print('embedding W created winth %d rnad inits'%rand_count)
    return embeddings_W

# embeddings_W = create_embeddings_W(embeddings_dict, vocab)
# np.save('embeddings_W.npy', embeddings_W)

#embeddings_W = np.load('embeddings_W_200d.npy')
#print('embeddings_W loaded')

### MODEL ###
def linear_layer(inputs, num_outputs):
    W = tf.truncated_normal([tf.shape(inputs)[1], num_outputs], stddev=0.1)
    b = tf.constant(0.1, shape=[num_outputs])
    return tf.matmul(inputs, W)


## PLACEHOLDERS
story = tf.placeholder(tf.int64, [None, None, None], "story")  # [batch_size x 5 x max_length]
sentence_len = tf.placeholder(tf.int64, [None, None], "sentence_len")  # [batch_size x 5]
order = tf.placeholder(tf.int64, [None, None], "order")  # [batch_size x 5]
keep_prob = tf.placeholder(tf.float32)

batch_size = tf.shape(story)[0]

sentences = [tf.reshape(x, [batch_size, -1]) for x in tf.split(1, 5, story)]  # 5 times [batch_size x max_length]

# Word embeddings
# embeddings = tf.get_variable("W", [vocab_size, input_size], trainable=True,
#                              dtype=tf.float32)
# embeddings = embeddings.assign(embeddings_W)
embeddings = tf.Variable(
    tf.random_uniform([vocab_size, input_size], -1.0, 1.0), name="W")

sentences_embedded = [tf.nn.embedding_lookup(embeddings, sentence)  # [batch_size x max_seq_length x input_size]
                      for sentence in sentences]

lstm = LayerNormalizedLSTMCell(lstm_size)
lstm_dropout = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_dropout] * n_rnn_layers, state_is_tuple=True)
hs = []
with tf.variable_scope("encoder") as varscope:
    for i in range(5):
        # START OF BIDIRECTIONAL - not in final model since it did not improve performance
        # _,states  = tf.nn.bidirectional_dynamic_rnn(stacked_lstm,stacked_lstm, sentences_embedded[i], sequence_length=sentence_len[:,i], dtype=tf.float32)
        # rnn_final_state_fw, rnn_final_state_bw = states
        # hs.append(rnn_final_state_fw[-1].h)
        # hs.append(rnn_final_state_bw[-1].h)
        # varscope.reuse_variables()
        # END OF BIDIRECTIONAL
        _, rnn_final_state = tf.nn.dynamic_rnn(stacked_lstm, sentences_embedded[i],
                                               sequence_length=sentence_len[:, i],
                                               dtype=tf.float32)
        hs.append(rnn_final_state[-1].h)
        varscope.reuse_variables()

# START OF SEQ2SEQ ATTENTION ADD-ON  (not in final model)
# seq2seq
# lstm = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
# stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * n_rnn_layers,
#                                           state_is_tuple=True)

# encoder_inputs = hs
# one_hot = tf.one_hot(order, 5, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)
# pad with zeros for GO symbol
# one_hot = tf.pad(one_hot, [[0,0],[1,0],[0,0]])
# convert to list of inputs
# decoder_inputs = [tf.reshape(x, [batch_size, 5]) for x in tf.split(1, 6, one_hot)]
# num_decoder_symbols = 5
# outputs, _ = attention_seq2seq(encoder_inputs, decoder_inputs, stacked_lstm,
#                                num_decoder_symbols,num_heads=5,
#                                feed_previous=feed_previous)

# concat LSTM outputs and ignore the last one (EOS)
# logits_flat = tf.concat(1, outputs[:-1])    # [batch_size x 5*input_size]
# END OF SEQ2SEQ ATTENTION ADD-ON

# concat LSTM outputs
h = tf.concat(1, hs)  # [batch_size x 5*input_size]
h = tf.reshape(h, [batch_size, 5 * lstm_size])
# FOR BIDIRECTIONAL
# h = tf.reshape(h, [batch_size, 5 * lstm_size * 2 ])

# FULLY CONNECTED LAYERS (in final model)
# fc1 = tf.contrib.layers.fully_connected(h, fc_size)
# fc1_drop = tf.nn.dropout(fc1, keep_prob)
fc2 = tf.contrib.layers.fully_connected(h, fc_size)
fc2_drop = tf.nn.dropout(fc2, keep_prob)
fc3 = tf.contrib.layers.linear(fc2_drop, 5 * target_size)  # [batch_size x 5*target_size]
logits = tf.reshape(fc3, [-1, 5, target_size])  # [batch_size x 5 x target_size]

# loss
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))
# L2 regularisation attempt - not in final model
# loss = (tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))
#         + 0.01*tf.nn.l2_loss(h) + 0.01*tf.nn.l2_loss(fc2))

# prediction function
predict = tf.arg_max(logits, 2)
correct_prediction = tf.equal(predict, order)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# EXPERIMENTATION WITH OTHER TECHNIQUES, MODEL TRAINING and ERROR ANALYSIS OMITTED

# ASSESSMENT

# LOAD THE DATA
data_test = nn.load_corpus(data_path + "dev.tsv")
# make sure you process this with the same pipeline as you processed your dev set
test_stories, test_stories_len, test_orders, _ = pipeline(data_test, vocab=vocab,
                                                          max_sent_len_=max_sent_len)

# THIS VARIABLE MUST BE NAMED `test_feed_dict`
test_feed_dict = {story: test_stories, sentence_len: test_stories_len,
                  order: test_orders, keep_prob: 1.0}

# ! ASSESSMENT 1 - DO NOT CHANGE, MOVE NOR COPY
with tf.Session() as sess:
    # LOAD THE MODEL
    saver = tf.train.Saver()
    saver.restore(sess, './model/model.checkpoint')

    # RUN TEST SET EVALUATION
    dev_predicted = sess.run(predict, feed_dict=test_feed_dict)
    dev_accuracy = nn.calculate_accuracy(dev_orders, dev_predicted)

dev_accuracy























































