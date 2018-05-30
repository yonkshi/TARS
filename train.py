from __future__ import absolute_import, division, print_function
import conf
from dataloader import DataLoader, index2str
import datetime
from model import *
import numpy as np


def main():
    # Hyper params
    update_steps = 100000
    learning_rate = 1e-4
    # Data path
    csv_filepath = 'asset/data/preprocess-librispeech/meta/train.csv'

    data = DataLoader(batch_size=conf.BATCH_SIZE, csv_filepath=csv_filepath)
    labels, label_text, lab_len, x, seq_length_col, x_file_name = data.training_set()

    seq_length = tf.reshape(seq_length_col, [-1]) # 1-D vector
    lab_len = tf.reshape(lab_len, [-1])  # 1-D vector

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate) # Try different gradient

    grads_vars, loss, wavenet_out = grad_tower(opt, labels, lab_len, x, seq_length)

    opt_op = opt.apply_gradients(grads_vars)
    print('model built')
    mean_loss = tf.reduce_mean(loss)
    loss_summary_op = tf.summary.scalar('loss', mean_loss, family='loss and accuracy')

    summary_op = tf.summary.merge([loss_summary_op])
    run_name = datetime.datetime.now().strftime("May_%d_%I_%M%p")
    writer = tf.summary.FileWriter('./tb_logs/%s' % run_name)
    saver = tf.train.Saver()

    # DEBUGING STUFF
    #densified_label = tf.sparse_tensor_to_dense(labels)
    #dense_predicted = tf.sparse_tensor_to_dense(predicted_out[0])

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(data.iterator.initializer)

        for step in range(update_steps):

            if step % 10 == 0:
                #a,b,c,d, = sess.run([labels]) debugging pipeline output
                _, loss_out, summary, _x_out_, _wavenet_out_, _label_text_, _seq_len, _x_file_name = sess.run([opt_op, loss, summary_op, x, wavenet_out, label_text, seq_length, x_file_name])
                print('step',step,'loss', np.mean(loss_out))
                writer.add_summary(summary, step)

                filename = _x_file_name[0].decode('utf-8')
                label_idx = np.fromstring(_label_text_[0], np.int64)
                label = index2str(label_idx)
                predicted, _ = keras.backend.ctc_decode(_wavenet_out_[0:1, :, :], _seq_len[0:1], greedy=False)
                _predicted = sess.run([predicted])
                _predicted = index2str(_predicted[0][0][0])
                print('labels   :',label)
                print('predicted:',_predicted)
                print('filename:', filename)
                print('')

                #print(_wavenet_out_)

            else:
                _ = sess.run([opt_op])

            if step % 10000 == 0 and step > 0:
                saver.save(sess, 'saved/%s' % run_name)



def grad_tower(opt, labels, lab_len, x, seq_length):
    # Build model
    with tf.device('/gpu:0'):
        wavenet_out, wavenet_no_softmax = build_wavenet(x, voca_size=conf.ALPHA_SIZE)
        loss  = tf.keras.backend.ctc_batch_cost(labels, wavenet_out, seq_length, lab_len)
        loss_mean = tf.reduce_mean(loss)
        wavenet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wavenet')
        grads_vars = opt.compute_gradients(loss_mean, wavenet_weights)

        return grads_vars, loss, wavenet_out


def calc_accuracy(labels, wavenet_out, seq_len ):

    wavenet_timemajor = tf.transpose(wavenet_out,[1,0,2]) # Time major for ctc
    predicted_out, _ = tf.nn.ctc_beam_search_decoder(wavenet_timemajor, seq_len, merge_repeated=False)
    # to dense tensor
    p = tf.sparse_to_dense(predicted_out[0].indices, predicted_out[0].dense_shape, predicted_out[0].values)
    y = tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values)
    p = tf.cast(p, dtype=tf.int32, name='casted_p')

    y_len = tf.shape(y)[1]
    p_len = tf.shape(p)[1]
    pad_dim = tf.abs(tf.shape(y)[1] - tf.shape(p)[1])

    y, p = tf.cond(y_len > p_len, lambda: (y, tf.pad(p, [[0, 0], [0, pad_dim]])), lambda:(tf.pad(y, [[0, 0], [0, pad_dim]]), p))

    d = p - y
    accuracy = (1 - tf.count_nonzero(d, dtype=tf.int32) / tf.size(y)) * 100

    return accuracy, predicted_out
if __name__ == '__main__':
    main()
