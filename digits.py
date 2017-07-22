"""
__name__ = digits.py
__author__ = Yash Patel
__description__ = Most basic use of deep learning, in the context of MNIST for
recognizing sudoku digits
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def basic_model(mnist):
    x = tf.placeholder(tf.float32, [None, 784])

    W1 = tf.Variable(tf.zeros([784, 10]))
    b1 = tf.Variable(tf.zeros([10]))

    y  = tf.matmul(x, W1) + b1
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    eta = 0.05
    train_step = tf.train.GradientDescentOptimizer(eta).minimize(cross_entropy)
    
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    TIME_STEPS = 1000
    for i in range(TIME_STEPS):
        print("Completed step: {}".format(i * TIME_STEPS))
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            x  : batch_xs,
            y_ : batch_ys
         })

        print(sess.run(accuracy, feed_dict={
            x  : mnist.test.images,
            y_ : mnist.test.labels
        }))

def standard_model(mnist):
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    x_img = tf.reshape(x, [-1, 28, 28, 1])

    W1 = weight_variable([5, 5, 1, 32])
    b1 = bias_variable([32])
    h1 = tf.nn.relu(conv2d(x_img, W1) + b1)
    h1_pool = max_pool_2x2(h1)

    W2 = weight_variable([5,5,32,64])
    b2 = bias_variable([64])
    h2 = tf.nn.relu(conv2d(h1, W2) + b2)
    h2_pool = max_pool_2x2(h2)

    h2_pool_trans = tf.reshape(h2_pool, [-1, 7 * 7 * 64])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h2_pool_trans, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    dropout = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y  = tf.matmul(dropout, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={
                x  : batch_xs,
                y_ : batch_ys,
                keep_prob : .80
            })

if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    standard_model(mnist)