import tensorflow as tf
from wavenet.data import SpeechCorpus, voca_size
from wavenet.model import *
import wavenet.conf as conf



__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
#tf.sg_verbosity(10)


#
# hyper parameters
#
max_episode = 15
learning_rate = 1e-4
#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=conf.BATCH_SIZE)


# # mfcc feature of audio queue
# inputs = data.mfcc
# # target sentence label queue
# labels = data.mfcc

# sequence length except zero-padding
# seq_len = []
# for input_ in inputs:
#     input_ = tf.reduce_sum(input_, axis=2)
#     input_ = tf.not_equal(input_, 0.)
#     input_ = tf.cast(input_, 'int')
#     input_ = tf.reduce_sum(input_, axis=1)
#     seq_len.append(input_)


# parallel loss tower
y_batch, x_batch, seq_length_col = data.next_batch
wavenet_out = get_model(x_batch, voca_size=voca_size)
seq_length = tf.transpose(seq_length_col)

wavenet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wavenet')

#loss = tf.nn.ctc_loss(wavenet_out, y_batch, seq_length)


#
# train
#
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_writer = tf.summary.FileWriter('./logs/1/train ', sess.graph)
    sess.run(data.iterator.initializer)

    for i in range(10):
        out1, out2 = sess.run([wavenet_out, y_batch])
        print('yo')


#tf.sg_train(lr=0.0001, loss=loss,
#            ep_size=data.num_batch, max_ep=50)
