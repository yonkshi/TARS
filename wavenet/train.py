from __future__ import absolute_import, division, print_function
import datetime

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np

from wavenet.data import SpeechCorpus, voca_size
from wavenet.model import *
import wavenet.conf as conf

max_episode = 15
learning_rate = 1e-4
# corpus input tensor
data = SpeechCorpus(batch_size=conf.BATCH_SIZE)

# parallel loss tower
y_batch, x_batch, seq_length_col = data.next_batch

seq_length = tf.reshape(seq_length_col, [-1]) # 1-D vector


# Build model
wavenet_out = get_model(x_batch, voca_size=voca_size)
loss = tf.nn.ctc_loss(y_batch, wavenet_out, seq_length, time_major=False) # predicted = batch x time x feat_dim
opt = tf.train.AdamOptimizer() # Trying something
wavenet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wavenet')
grads_vars = opt.compute_gradients(loss, wavenet_weights)

opt_op = opt.apply_gradients(grads_vars)
print('model built')

run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p")
train_writer = tf.summary.FileWriter('./logs/%s' % run_name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(data.iterator.initializer)

    for i in range(100):
        seqlenlenelne = sess.run(seq_length)
        print('training:',i)
        _, loss_out = sess.run([opt_op, loss])
        print('loss',np.mean(loss_out))
