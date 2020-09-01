# Day_07_01_RnnBasic3.py
import tensorflow as tf
import numpy as np


def show_sequence_loss(targets, logits):
    y = tf.constant(targets)
    z = tf.constant(logits)

    w = tf.ones([1, len(targets[0])])

    loss = tf.contrib.seq2seq.sequence_loss(logits= z, targets=y, weights=w)
    # loss = tfa.seq2seq.sequence_loss(logits= z, targets=y, weights=w)

    sess = tf.Session()
    print(sess.run(loss), targets, logits)
    sess.close()


pred1 = [[[0.2, 0.7], [0.6, 0.4], [0.1, 0.5]]]
#pred2 = [[[0.7, 0.2], [0.4, 0.6], [0.5, 0.1]]]

show_sequence_loss([[1, 1, 1]], pred1)
#show_sequence_loss([[0, 0, 0]], pred2)

#show_sequence_loss([[1,1,1]],pred1)
show_sequence_loss([[2,2,2]],[[[0.2, 0.7, 0.1], [0.6, 0.4, 0.0], [0.4, 0.1, 0.5]]])

