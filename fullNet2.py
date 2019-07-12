#coding=utf-8
#Version:python3.6.0
#搭建神经网络八股：准备 前传 反传 迭代
#前传 x y_ w1 w2 a y
#反传 loss train_step
#迭代 sess
#损失函数loss
#学习率learning_rate
#滑动平均ema
#正则化regularization
_date_ = '2019/7/10 9:38'
_author_ = 'tan'
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455
#基于seed产生随机数
rng = np.random.RandomState(seed)
X = rng.rand(32, 2)
#从X中取一行，如果和小于1就给Y赋值1，，如果不小于1 赋0
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]
print(X)
print(Y)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
#均方根误差
loss = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(w1))
    print(sess.run(w2))

    #训练模型
    STEPS = 3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x:X, y_:Y})
            print("After %d training step(s),loss on all data is %g" %(i, total_loss))
    #训练后的参数
    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))