from __future__ import absolute_import, division, print_function
import datetime

from tensorflow.python import debug as tf_debug
import numpy as np

from dataloader import DataLoader, index2str
from model import *
import conf as conf

def main():
    # Hyper params
    update_steps = 100_000
    learning_rate = 1e-3

    # Data pipeline initialization
    data = DataLoader(batch_size=conf.BATCH_SIZE)
    labels, label_text, x, seq_length_col, x_file_name = data.training_set()

    # Bug fix because seq_length does not support
    seq_length = tf.reshape(seq_length_col, [-1]) # 1-D vector

    # optimizer
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate) # Try different gradient

    # forward and backpass inside grad_tower, meant for single GPU operation
    grads_vars, loss, wavenet_out = grad_tower(opt, labels, x, seq_length)

    # Apply gradients to models
    opt_op = opt.apply_gradients(grads_vars)
    print('model built')


    # Summary related stuff, not essential to op
    mean_loss = tf.reduce_mean(loss)
    loss_summary_op = tf.summary.scalar('loss', mean_loss, family='loss and accuracy')
    accuracy_op, predicted_out = calc_accuracy(labels,wavenet_out, seq_length)
    accuracy_summary_op = tf.summary.scalar('accuracy', accuracy_op, family='loss and accuracy')

    summary_op = tf.summary.merge([accuracy_summary_op, loss_summary_op])
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p")
    writer = tf.summary.FileWriter('./tb_logs/%s' % run_name)
    saver = tf.train.Saver()

    # DEBUGING STUFF
    densified_label = tf.sparse_tensor_to_dense(labels)
    dense_predicted = tf.sparse_tensor_to_dense(predicted_out[0])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(data.iterator.initializer)

        for step in range(update_steps):
            # compute summary every 10 steps
            if step % 10 == 0:
                _, loss_out, accuracy_out,summary, _x_out_, _wavenet_out_, _label_text_, _densified_label_, _seq_len, _x_file_name, _predicted_out = sess.run([opt_op, loss, accuracy_op, summary_op, x, wavenet_out, label_text, densified_label, seq_length, x_file_name, dense_predicted])
                print('step', step, 'loss', np.mean(loss_out))
                writer.add_summary(summary, step)

                label_idx = np.fromstring(_label_text_[0], np.int64)
                label = index2str(label_idx)
                predicted = index2str(_predicted_out[0])
                print('labels   :', label)
                print('predicted:', predicted)

                # Everything below are used for debugging loss
                for i in range(conf.BATCH_SIZE):
                    if loss_out[i] == 0.0:
                        print('>>>>>>> zero loss')

                        print('loss', loss_out[i])

                        __x_out = _x_out_[i]
                        __wavenet_out = _wavenet_out_[i]
                        __label_text_ = _label_text_[i]
                        __densified_label_ = _densified_label_[i]
                        __seq_len = _seq_len[i]
                        __predicted_out = _predicted_out[i]

                        filename = _x_file_name[i].decode('utf-8')
                        label_idx = np.fromstring(_label_text_[i], np.int64)
                        label = index2str(label_idx)
                        predicted = index2str(_predicted_out[i])

                        print('labels   :', label)
                        print('predicted:', predicted)

                        print('label length', len(label))

                        print('filename:', filename)

                        print('sequence_len:', _seq_len[i])
                        print('_wavenet_out_: ', _wavenet_out_[i])


                        print('end <<<<')
                        print('>>>>>\n\n\n\n')

            else:
                _ = sess.run([opt_op])

            if step % 10_000 == 0 and step > 0:
                saver.save(sess, 'saved/%s' % run_name)



def grad_tower(opt, labels, x, seq_length):
    # Build model
    with tf.device('/gpu:0'):
        # Forward pass
        wavenet_out, wavenet_no_softmax = build_wavenet(x, voca_size=conf.ALPHA_SIZE)

        # Loss function
        loss = tf.nn.ctc_loss(labels, wavenet_no_softmax, seq_length,
                              time_major=False, # batch x time x alpha_dimgt
                              ctc_merge_repeated=False, # So we don't have to manually add <emp> at each repeating char
                              ignore_longer_outputs_than_inputs=True) # predicted = batch x time x feat_dim

        # Mean
        loss_mean = tf.reduce_mean(loss)

        # Get all wavenet parameters
        wavenet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wavenet')

        # Backprop
        grads_vars = opt.compute_gradients(loss_mean, wavenet_weights)

        return grads_vars, loss, wavenet_no_softmax


def calc_accuracy(labels, wavenet_out, seq_len ):
    # Used to visualize trainig progress
    wavenet_timemajor = tf.transpose(wavenet_out,[1,0,2]) # Time major for ctc
    predicted_out, _ = tf.nn.ctc_beam_search_decoder(wavenet_timemajor, seq_len, merge_repeated=False) # ,
    # to dense tensor

    # CTC Decoder
    p = tf.sparse_to_dense(predicted_out[0].indices, predicted_out[0].dense_shape, predicted_out[0].values)
    y = tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values)
    p = tf.cast(p, dtype=tf.int32, name='casted_p')

    # Computing character level difference
    # TODO Use edit distance instead of alignment distance
    y_len = tf.shape(y)[1]
    p_len = tf.shape(p)[1]
    pad_dim = tf.abs(tf.shape(y)[1] - tf.shape(p)[1])

    y, p = tf.cond(y_len > p_len, lambda: (y, tf.pad(p, [[0, 0], [0, pad_dim]])), lambda:(tf.pad(y, [[0, 0], [0, pad_dim]]), p))

    d = p - y
    accuracy = (1 - tf.count_nonzero(d, dtype=tf.int32) / tf.size(y)) * 100

    return accuracy, predicted_out
if __name__ == '__main__':
    main()