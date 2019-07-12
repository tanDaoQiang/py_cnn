#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/10 10:32'
_author_ = 'tan'
#自定义损失函数
#w1 = w0 - learning_rate*梯度
#滑动平均：记录每个参数在一段时间内过往值得平均，增加模型的泛化性
#正则化：
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
seed = 23455
COST = 1
PROFIT = 9
#基于seed产生随机数
rng = np.random.RandomState(seed)
X = rng.rand(32, 2)
#从X中取一行，如果和小于1就给Y赋值1，，如果不小于1 赋0
Y = [[x1+x2+(rng.rand()/10.0-0.05)] for (x1, x2) in X]
print(X)
print(Y)

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)
#均方根误差
loss = tf.reduce_mean(tf.where(tf.greater(y, y_), (y - y_)*COST, (y_ - y)*PROFIT))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(w1))

    #训练模型
    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            print("After %d training step(s),w1 is " %(i))
            print(sess.run(w1))
    #训练后的参数
    print("\n")
    print("w1:\n", sess.run(w1))