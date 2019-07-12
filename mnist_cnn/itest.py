#coding=utf-8
#Version:python3.6.0
_date_ = '2019/7/10 15:58'
_author_ = 'tan'

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn.forward as forword
import mnist_cnn.backward as backward

TEST_INTERVAL_SECS = 5
def itest(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, forword.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, forword.OUTPUT_NODE])
        y = forword.forward(x, None)

        #实例化带滑动平均的saver对象
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        #计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                #加载ckpt，把滑动平均值赋给每个参数
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #恢复模型到当前会话
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #恢复global_step
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
                    print("After %s training steps. test accuracy =%g"%(global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)
def main():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    itest(mnist)

if __name__ == '__main__':
    main()