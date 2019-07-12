#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/10 16:27'
_author_ = 'tan'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn.forward as forword
import os

STEPS = 50000
BATCH_SIZE = 200
#最开始的学习率
LEARNNING_RATE_BASE = 0.1
#学习率衰减率
LEARNNING_RATE_DECAY = 0.99
#正则化系数
REGULARIZER = 0.0001
#滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

def backward(mnist):
    x = tf.placeholder(tf.float32, [None, forword.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, shape=(None, forword.OUTPUT_NODE))
    y = forword.forward(x, REGULARIZER)
    #轮数计数器设定为不可训练
    global_step = tf.Variable(0, trainable=False)
    #调用包含正则化的损失函数
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    #指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNNING_RATE_DECAY,
        staircase=True
    )

    #定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')
    #实例化saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        #给所有的w b赋值
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            # 恢复模型到当前会话
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs,y_: ys})

            if i % 1000 == 0:
                print("after %d training steps,loss on training batch is: %g" %(step,loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME), global_step=global_step)
def main():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    backward(mnist)

if __name__ == '__main__':
    main()
