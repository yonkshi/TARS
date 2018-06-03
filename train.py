import datetime
import tensorflow as tf
import numpy as np
from dataloader import DataLoader, index2str
from model import build_wavenet, what_the_hell
import conf as conf

def main():
    # Hyper params
    update_steps = 100_000
    learning_rate = 1e-3
    num_gpu = conf.NUM_GPU

    data = DataLoader(batch_size=conf.BATCH_SIZE)
    labels, label_text, x, seq_length_col, x_file_name = data.training_set()

    seq_length = tf.reshape(seq_length_col, [-1]) # 1-D vector

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate) # Try different gradient

    #grads_vars, loss, wavenet_out = grad_tower(opt, labels, x, seq_length)

    # Multi GPU support
    def split_tensor_for_gpu(t):
        return tf.split(t, num_gpu)

    # Splitting batch for multiple GPUs
    x_gpus = split_tensor_for_gpu(x)
    seq_length_gpus = split_tensor_for_gpu(seq_length)
    labels_gpu = tf.sparse_split(sp_input=labels, num_split=num_gpu, axis=0)

    # TODO double splitted values
    grads = []
    loss = 0
    for i in range(num_gpu):
        # Runs on GPU

        grads_vars_gpu, loss_gpu, wavenet_out = grad_tower(i, opt, labels_gpu[i], x_gpus[i], seq_length_gpus[i])

        # normalize and element wise add
        if not grads:
            grads = [g_grad / num_gpu for g_grad, g_vars in grads_vars_gpu]
        else:
            # Element wise add to G_grads collection, G_grads is same size as G_grads_vars' grads
            grads = [g_grad / num_gpu + grads[j] for j, (g_grad, g_vars) in enumerate(grads_vars_gpu)]

        loss = loss_gpu / num_gpu + loss


    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wavenet')

    opt_op = opt.apply_gradients(zip(grads, vars))

    # Stuff for summary/logging, not really essential to implementation
    print('model built')
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

        # Main loop
        for step in range(update_steps):
            # Print out summary every 10 steps
            if step % 10 == 0:
                #a,b,c,d, = sess.run([labels]) debugging pipeline output
                _, loss_out, accuracy_out,summary, _x_out_, _wavenet_out_, _label_text_, _densified_label_, _seq_len, _x_file_name, _predicted_out = sess.run([opt_op, loss, accuracy_op, summary_op, x, wavenet_out, label_text, densified_label, seq_length, x_file_name, dense_predicted])
                print('step', step, 'loss', np.mean(loss_out))
                writer.add_summary(summary, step)
                label_idx = np.fromstring(_label_text_[0], np.int64)
                label = index2str(label_idx)
                predicted = index2str(_predicted_out[0])
                print('labels   :', label)
                print('predicted:', predicted)

                # Debugging mean 0 issue!
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



def grad_tower(gpu_num, opt, labels, x, seq_length):
    # Build model
    with tf.device('/gpu:%d' % gpu_num):
        with tf.name_scope('gpu_ZZZZZZZ_%d' % gpu_num):
            reuse = gpu_num != 0 # initalize first wavenet
            wavenet_out = what_the_hell(x, voca_size=conf.ALPHA_SIZE, reuse=reuse)

            # loss = tf.nn.ctc_loss(labels, wavenet_no_softmax, seq_length,
            #                       time_major=False, # batch x time x alpha_dimgt
            #                       ctc_merge_repeated=False, # So we don't have to manually add <emp> at each repeating char
            #                       ignore_longer_outputs_than_inputs=True) # predicted = batch x time x feat_dim
            loss_mean = tf.reduce_mean(wavenet_out)
            all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            wavenet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='wavenet')
            grads_vars = opt.compute_gradients(loss_mean, wavenet_vars)

            return grads_vars, loss, wavenet_no_softmax


def calc_accuracy(labels, wavenet_out, seq_len ):

    wavenet_timemajor = tf.transpose(wavenet_out,[1,0,2]) # Time major for ctc
    predicted_out, _ = tf.nn.ctc_beam_search_decoder(wavenet_timemajor, seq_len, merge_repeated=False) # ,
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