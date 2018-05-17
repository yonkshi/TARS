import sugartensor as tf
from wavenet.data import SpeechCorpus, voca_size
from wavenet.model import *


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
#tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # total batch size

#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=batch_size)

# mfcc feature of audio
inputs = data.mfcc
# target sentence label
labels = data.mfcc

# sequence length except zero-padding
seq_len = []
for input_ in inputs:
    input_ = tf.reduce_sum(input_, axis=2)
    input_ = tf.not_equal(input_, 0.)
    input_ = tf.cast(input_, 'int')
    input_ = tf.reduce_sum(input_, axis=1)
    seq_len.append(input_)


# parallel loss tower
@tf.sg_parallel
def get_loss(opt):
    # encode audio feature
    logit = get_logit_keras(opt.input[opt.gpu_index], voca_size=voca_size)
    # CTC loss
    return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])

#
# train
#
tf.sg_train(lr=0.0001, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
            ep_size=data.num_batch, max_ep=50)
