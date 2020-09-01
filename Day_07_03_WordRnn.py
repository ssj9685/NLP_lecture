import tensorflow as tf
import numpy as np

def make_data(origin):
    word = sorted(set(origin))

    chr2idx = {ch:i for i, ch in enumerate(word)}

    word_idx = [chr2idx[t] for t in origin]
    x = word_idx[:-1]
    y = word_idx[1:]

    eye = np.eye(len(word),dtype=np.int32)

    x = eye[x]
    x = np.float32([x])
    y = np.int32([y])
    y = tf.constant(y)

    return x, y, np.array(word)

def rnn_word(word, n_iteration=100):
    x, y, vocab = make_data(word)

    batch_size, seq_length, n_classes = x.shape

    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    
    z = tf.layers.dense(inputs=outputs,units=n_classes,activation=None)

    w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(logits= z, targets=y, weights=w)
    # loss = tfa.seq2seq.sequence_loss(logits= z, targets=y, weights=w)

    sess = tf.Session()
    sess.close()
    
    hx = tf.nn.softmax(z)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(n_iteration):
        sess.run(train)
        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2)
        preds_arg = preds_arg.reshape(-1)
        if(i%5==0):
            print(i, sess.run(loss), preds_arg,''.join(vocab[preds_arg]))
    sess.close()

rnn_word('deep learning',n_iteration=300)