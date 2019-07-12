#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/10 9:20'
_author_ = 'tan'
import tensorflow as tf
#常量
#x = tf.constant([[0.7, 0.5]])
#占位符
#x = tf.placeholder(tf.float32, shape=(1, 2))
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3]))
w2 = tf.Variable(tf.random_normal([3, 1]))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    #print(sess.run(y))
    #print(sess.run(y, feed_dict={x: [[0.7, 0.5]]}))
    print(sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))