#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/9 9:34'
_author_ = 'tan'
##使用卷积神经网络提取图片轮廓
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import tensorflow as tf
iimg = img.imread('img.jpg')
plt.imshow(iimg)
plt.axis('off')
plt.show()
print(iimg.shape)
#sobel算子
#shape[batch,in_height,in_weight,in_channels]
full = np.reshape(iimg,[1, 3268, 3268, 3])
inputfull = tf.Variable(tf.constant(1.0,shape=[1, 3268, 3268, 3]))
filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                 [-2.0,-2.0,-2.0],[0,0,0],[2.0,2.0,2.0],
                                 [-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0]],
                                 shape=[3,3,3,1]))
op = tf.nn.conv2d(inputfull,filter,strides=[1,1,1,1],padding='SAME')
o = tf.cast(((op -tf.reduce_mean(op))/(tf.reduce_max(op) - tf.reduce_min(op)))*255,tf.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t, f = sess.run([o, filter],feed_dict={inputfull:full})
    t =np.reshape(t,[3268,3268])
    plt.imshow(t)
    plt.axis('off')
    plt.show()
