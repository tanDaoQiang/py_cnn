#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/10 16:27'
_author_ = 'tan'
import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
import numpy as np
import modularization_cnn.generateds as generateds
import modularization_cnn.forward as forward

STEPS = 40000
BATCH_SIZE = 30
LEARNNING_RATE_BASE = 0.001
LEARNNING_RATE_DECAY = 0.999
REGULARIZER = 0.01

def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    X, Y_, Y_c = generateds.generateds()

    y = forward.forward(x, REGULARIZER)

    global_step = tf.Variable(0, trainable=False)
    #指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNNING_RATE_BASE,
        global_step,
        300/BATCH_SIZE,
        LEARNNING_RATE_DECAY,
        staircase=True
    )

    #定义损失函数
    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

    #定义反向传播方法：包含正则化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 300
            end = start +BATCH_SIZE
            sess.run(train_step, feed_dict={x:X[start:end], y_:Y_[start:end]})
            if i % 2000 == 0:
                loss_v = sess.run(loss_total, feed_dict={x:X,y_:Y_})
                print("after %d steps,loss is: %f" %(i,loss_v))
            #在-3到3生成网格做标点
            xx, yy = np.mgrid[-3:3:.01, -3:3:.01]
            #组成坐标集
            grid = np.c_[xx.ravel(), yy.ravel()]
            #把grid放入神经网络
            probs = sess.run(y, feed_dict={x:grid})
            probs = probs.reshape(xx.shape)

    plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()

if __name__ == '__main__':
    backward()
