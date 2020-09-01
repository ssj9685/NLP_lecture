import tensorflow as tf
import numpy as np
from sklearn import preprocessing

def make_data_1():
    x = [[0.,0.,0.,0.,0.,1.], #t
        [1.,0.,0.,0.,0.,0.], #e
        [0.,1.,0.,0.,0.,0.], #n
        [0.,0.,0.,0.,1.,0.], #s
        [0.,0.,1.,0.,0.,0.]] #o

    y = [0,1,4,2,3] #ensor

    x = np.float32([x])
    y = np.int32([y])
    y = tf.constant(y)

    return x, y

def make_data_2(origin):
    word = sorted(set(origin))

    idx2chr = {i:ch for i, ch in enumerate(word)}
    chr2idx = {ch:i for i, ch in enumerate(word)}

    word_idx = [chr2idx[t] for t in origin]
    x = word_idx[:-1]
    y = word_idx[1:]

    eye = np.eye(len(word),dtype=np.int32)

    x = eye[x]
    x = np.float32([x])
    y = np.int32([y])
    y = tf.constant(y)

    return x, y, np.array(idx2chr)

def make_data_3(origin):
    lb = preprocessing.LabelBinarizer()
    word = lb.fit_transform(list(origin))
    x = np.float32([word[:-1]])
    y = np.argmax(word[1:],axis=1)
    y = tf.constant(np.int32([y]))
    return x, y, lb.classes_

def rnn4(word):
    x, y, vocab = make_data_3(word)

    batch_size, seq_length, n_classes = x.shape
    print(x.shape)
    hidden_size = 7
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    z = tf.layers.dense(inputs=outputs,units=n_classes,activation=None)

    w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(logits= z, targets=y, weights=w)
    # loss = tfa.seq2seq.sequence_loss(logits= z, targets=y, weights=w)

    sess = tf.Session()
    sess.close()

    w = tf.Variable(tf.random.uniform([hidden_size, 6]))
    b = tf.Variable(tf.random.uniform([6]))

    # (5,6) = (5,7) @ (7,6)
    
    hx = tf.nn.softmax(z)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        preds = sess.run(z)
        preds_arg = np.argmax(preds, axis=2)
        preds_arg = preds_arg.reshape(-1)
        if(i%5==0):
            print(i, sess.run(loss),np.argmax(sess.run(z),axis=1),''.join(vocab[preds_arg]))
    sess.close()

def rnn4_3d():
    a = np.array([[0,1],[2,3],[4,5]])
    b = np.array([[0,1,2],[3,4,5]])

    ta = tf.Variable(a)
    tb = tf.Variable(b)

    taa = tf.Variable([a,a])
    tbb = tf.Variable([b,b])

    sess = tf.Session()
    hx = sess.run(tf.global_variables_initializer())
    hx = sess.run(tf.matmul(ta,tb))

    print(hx)
    print(hx.shape)

    sess.close()

rnn4('tensor')