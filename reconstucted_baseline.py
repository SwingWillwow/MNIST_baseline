import tensorflow as tf
import numpy as np
import time
import datetime
from pathlib import Path
from tensorflow.examples.tutorials.mnist import input_data


model_num = str(input('model number?'))
CHECK_POINT_DIR = 'model/'+model_num+'/'
LOG_FREQUENCY = 1  # how often log training information
BATCH_SIZE = 128     # batch size
EPOCH_SIZE = int(60000 // BATCH_SIZE)
EPOCH_NUM = 100
MAX_STEPS = EPOCH_SIZE*EPOCH_NUM
TRAIN_LOG_DIR = 'train/'+model_num+'/'
DATA_DIR = str(Path.home())+'/training_data/MNIST-data/'
EVAL_LOG_DIR = 'eval/'+model_num+'/'


def activations_summaries(activations):
    tf.summary.histogram(activations.op.name + '/activations', activations)
    tf.summary.scalar(activations.op.name + '/sparsity', tf.nn.zero_fraction(activations))


def weight_variables(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial, name=name+'_weight')


def bias_variables(shape, name):
    initial = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name+'_bias')


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def load_data():
    mnist = input_data.read_data_sets(DATA_DIR, validation_size=0)
    # get training data
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # get evaluating data
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # init data set
    train_data_set = tf.data.Dataset().from_tensor_slices((train_data, train_labels))
    train_data_set = train_data_set.repeat(EPOCH_NUM).batch(BATCH_SIZE).shuffle(10000)
    eval_data_set = tf.data.Dataset().from_tensors((eval_data, eval_labels))

    return train_data_set, eval_data_set


def inference(images, keep_rate):
    # first conv layer
    reshaped_image = tf.reshape(images, shape=[-1, 28, 28, 1])
    W_conv1 = weight_variables([5, 5, 1, 32], name='conv1')
    b_conv1 = bias_variables([32], name='conv1')
    h_conv1 = tf.nn.relu(conv2d(reshaped_image, W_conv1) + b_conv1)
    activations_summaries(h_conv1)
    # first pooling

    h_pool1 = max_pooling_2x2(h_conv1)

    # second conv layer

    W_conv2 = weight_variables([5, 5, 32, 64], name='conv2')
    b_conv2 = bias_variables([64], name='conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    activations_summaries(h_conv2)
    # second pooling layer

    h_pool2 = max_pooling_2x2(h_conv2)

    # flatten the pooling result for the dense layer

    flatten_pooling = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # fully-connected layer 1
    W_fc1 = weight_variables([7 * 7 * 64, 1024], name='fc1')
    b_fc1 = bias_variables([1024], name='fc1')

    h_fc1 = tf.nn.relu(tf.matmul(flatten_pooling, W_fc1) + b_fc1)
    activations_summaries(h_fc1)
    # dropout

    h_fc1_after_drop = tf.nn.dropout(h_fc1, keep_rate)

    # fully-connected layer 2 (output layer)

    W_fc2 = weight_variables([1024, 10], name='fc2')
    b_fc2 = bias_variables([10], name='fc2')
    h_fc2 = tf.matmul(h_fc1_after_drop, W_fc2) + b_fc2
    activations_summaries(h_fc2)

    return h_fc2


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return cross_entropy_mean


def loss_summary(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9)
    loss_averages_op = loss_averages.apply([total_loss])
    tf.summary.scalar(total_loss.op.name + ' (raw)', total_loss)
    tf.summary.scalar(total_loss.op.name, loss_averages.average(total_loss))
    return loss_averages_op


def train(total_loss, global_step):
    loss_averages_op = loss_summary(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(1e-4)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step)

    for v in tf.trainable_variables():
        tf.summary.histogram(v.op.name, v)

    for grad, v in grads:
        if grad is not None:
            tf.summary.histogram(v.op.name + '/gradient', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op('train_op')
    return train_op


def accuracy(logits, labels):
    predict = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels)
    right_num = tf.reduce_sum(tf.cast(predict, tf.int32))
    total_num = labels.shape[0]
    return tf.div(tf.cast(right_num, tf.float32), total_num)


def train_model():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_images')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_labels')
    keep_rate = tf.placeholder(dtype=tf.float32, shape=None, name='keep_rate')
    logits = inference(x, keep_rate)
    losses = loss(logits, y)
    global_step = tf.train.get_or_create_global_step()
    train_op = train(losses, global_step=global_step)
    train_data_set, _ = load_data()
    train_iterator = train_data_set.make_initializable_iterator()
    next_data = train_iterator.get_next()
    with tf.Session() as sess:
        # initialization
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iterator.initializer)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(logdir=TRAIN_LOG_DIR)
        merged_summaries = tf.summary.merge_all()
        for i in range(1, EPOCH_NUM * EPOCH_SIZE + 1):
            images, labels = sess.run(next_data)
            sess.run(train_op, feed_dict={x: images, y: labels, keep_rate: 0.5})
            if i % 100 == 0:
                ac_rate = accuracy(logits, labels)
                ac, current_loss = sess.run([ac_rate, losses], feed_dict={x: images, y: labels, keep_rate: 0.5})
                print(('%d' % i) + ' step: current accuracy ' + str(ac), ', current loss '+str(current_loss))
            if i % 1000 == 0 or i == EPOCH_NUM * EPOCH_SIZE:
                saver.save(sess, CHECK_POINT_DIR, global_step)
                global_step_value = tf.train.global_step(sess, global_step)
                summaries = sess.run(merged_summaries, feed_dict={x: images,
                                                                 y: labels})
                summary_writer.add_summary(summaries, global_step_value)


def eval_model():
    _, eval_data_set = load_data()
    eval_iterator = eval_data_set.make_initializable_iterator()
    next_eval_data = eval_iterator.get_next()

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='images')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
    keep_rate = tf.placeholder(dtype=tf.float32, shape=None, name='keep_rate')
    logits = inference(x, keep_rate)
    losses = loss(logits, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(eval_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('Can not find checkpoint files.')
            return
        images, labels = sess.run(next_eval_data)
        total_loss, ac_rate = sess.run([losses, accuracy(logits, labels)],
                                       feed_dict={x: images, y: labels, keep_rate: 1})
        print('testing finished!')
        print('final loss:' + str(total_loss))
        print('final accuracy' + str(ac_rate))
        merged_summaries = tf.summary.merge_all()
        summaries = sess.run(merged_summaries, feed_dict={x: images,
                                                          y: labels,
                                                          keep_rate: 1})
        summaries_writer = tf.summary.FileWriter(EVAL_LOG_DIR)
        summaries_writer.add_summary(summaries, global_step)


def main(args):
    train_or_eval = str(input('train or eval model?'))
    is_train = train_or_eval == 'train'
    if is_train:
        if tf.gfile.Exists(CHECK_POINT_DIR):
            is_new = str(input('new model?'))
            is_new = is_new == 'True'
            if is_new:
                tf.gfile.DeleteRecursively(CHECK_POINT_DIR)
                tf.gfile.MakeDirs(CHECK_POINT_DIR)
                train_model()
                return
            else:
                train_model()
                return
        tf.gfile.MakeDirs(CHECK_POINT_DIR)
        train_model()
        return
    else:
        if tf.gfile.Exists(EVAL_LOG_DIR):
            tf.gfile.DeleteRecursively(EVAL_LOG_DIR)
        tf.gfile.MakeDirs(EVAL_LOG_DIR)
        eval_model()


if __name__ == '__main__':
    tf.app.run()


""" 

            deprecated methods
 
 methods below are deprecated methods, preserved for further potential usage.
 
"""
# def train_model():
#     # import data
#     with tf.Graph().as_default():
#         global_step = tf.train.get_or_create_global_step()
#         train_data_set, eval_data_set = load_data()
#         train_iterator = train_data_set.make_one_shot_iterator()
#         next_train_data = train_iterator.get_next()
#         images, labels = next_train_data
#         logits = inference(images)
#         losses = loss(logits, labels)
#         train_op = train(losses, global_step)
#
#     # self-defined hook to track training information
#         class _LoggerHook(tf.train.SessionRunHook):
#             def begin(self):
#                 self._step = -1
#                 self.start_time = time.time()
#
#             def before_run(self, run_context):
#                 self._step += 1
#                 return tf.train.SessionRunArgs([losses])
#
#             def after_run(self,
#                           run_context,
#                           run_values):
#                 if self._step % LOG_FREQUENCY == 0:
#                     current_time = time.time()
#                     duration = current_time - self.start_time
#                     self.start_time = current_time
#                     loss_value = run_values.result
#                     example_per_second = LOG_FREQUENCY * BATCH_SIZE / duration
#                     sec_per_batch = float(duration / LOG_FREQUENCY)
#
#                     format_str = ('%s: step %d, loss=%.6f, accuracy=%.4f (%.1f example/sec'
#                                   '%.3f sec/batch)')
#                     print(format_str % (datetime.datetime.now(), self._step, loss_value,
#                                         accuracy(logits, labels), example_per_second, sec_per_batch))
#
#         with tf.train.MonitoredTrainingSession(
#                 checkpoint_dir=CHECK_POINT_DIR,
#                 hooks=[tf.train.StopAtStepHook(MAX_STEPS),
#                        tf.train.NanTensorHook(losses),
#                        _LoggerHook()],
#                 config=tf.ConfigProto(log_device_placement=None)) as mon_sess:
#             while not mon_sess.should_stop():
#                 mon_sess.run(train_op)