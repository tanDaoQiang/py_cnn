#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/10 16:16'
_author_ = 'tan'
import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
#隐藏层节点个数
LAYER1_NODE = 500
#定义神经网络的输入，参数和输出，定义前向传播
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape, stddev=0.1))
    #正则化
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2

    return y