# reference
# https://github.com/sugyan/tensorflow-mnist/blob/master/mnist/model.py


import tensorflow as tf

import input_data
import data_adder
#data = input_data.read_data_sets("/tmp/data/", one_hot=True)
data = input_data.read_data_sets('MNIST_data', one_hot=True)
data = data_adder.add_data(data)

# model
import model
with tf.variable_scope("convolutional"):
    x = tf.placeholder("float", [None, 784])
    keep_prob = tf.placeholder("float")
    y, variables = model.convolutional(x, keep_prob)

# train
#y_ = tf.placeholder("float", [None, 10])
y_ = tf.placeholder("float", [None, 11])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver(variables)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
        sess.run(train_step, feed_dict={
                 x: batch[0], y_: batch[1], keep_prob: 0.5})

    print sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels, keep_prob: 1.0})

    #path = saver.save(sess, "data/convolutional.ckpt")
    path = saver.save(sess, "data/convolutional_tmp.ckpt")
    print "Saved:", path
