import tensorflow as tf
import numpy as np

w = 'tensor'
print("".join(sorted(list(w))))

x = [[0.,0.,0.,0.,0,1.],
     [1.,0.,0.,0.,0.,0.],
     [0.,1.,0.,0.,0.,0.],
     [0.,0.,0.,0.,1.,0.],
     [0.,0.,1.,0.,0.,0.]]

y = [[1.,0.,0.,0.,0.,0.],
     [0.,1.,0.,0.,0.,0.],
     [0.,0.,0.,0.,1.,0.],
     [0.,0.,1.,0.,0.,0.],
     [0.,0.,0.,1.,0.,0.]]

w = tf.Variable(tf.random.uniform([6, 6]))
b = tf.Variable(tf.random.uniform([6]))

# (5,6) = (5,6) @ (6,6)
z = tf.matmul(x, w)
hx = tf.nn.softmax(z)
loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=z)
loss = tf.reduce_mean(loss_i)

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):
    sess.run(train)
    if(i%5==0):
        print(i, sess.run(loss))

preds = sess.run(hx)
print(preds)
print(y)

preds_arg = np.argmax(preds, axis=1)
y_arg = np.argmax(y, axis=1)
print(preds_arg)
print(y_arg)

equals = (preds_arg == y_arg)
print(equals)

print('acc :', np.mean(equals))
sess.close()