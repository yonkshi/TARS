import tensorflow as tf
import keras
num_dilation = 3     # dilated blocks
num_dim = 128      # latent dimension
num_filters = 5
use_skipped_output = True # flag to choose if use resnet
num_res_blocks = 3  # Number of res_net blocks
num_res_hidden_layers = 3 # Number of res net hidden layers
feature_size = 13 # Input dimension

#
# logit calculating graph using atrous convolution
#
def get_logit(x, voca_size):

    # residual block
    def res_block(tensor, size, rate, block, dim=num_dim):

        with sugar_tf.sg_context(name='block_%d_%d' % (block, rate)):

            # filter convolution
            conv_filter = tensor.sg_aconv1d(size=size, rate=rate, act='tanh', bn=True, name='conv_filter')

            # gate convolution
            conv_gate = tensor.sg_aconv1d(size=size, rate=rate,  act='sigmoid', bn=True, name='conv_gate')

            # output by gate multiplying
            out = conv_filter * conv_gate

            # final output
            out = out.sg_conv1d(size=1, dim=dim, act='tanh', bn=True, name='conv_out')

            # residual and skip output
            return out + tensor, out

    # expand dimension
    with sugar_tf.sg_context(name='front'):
        z = x.sg_conv1d(size=1, dim=num_dim, act='tanh', bn=True, name='conv_in')

    # dilated conv block loop
    skip = 0  # skip connections
    for i in range(num_dilation):
        for r in [1, 2, 4, 8, 16]:
            z, s = res_block(z, size=7, rate=r, block=i)
            skip += s
    # final logit layers
    with sugar_tf.sg_context(name='logit'):
        logit = (skip
                 .sg_conv1d(size=1, act='tanh', bn=True, name='conv_1')
                 .sg_conv1d(size=1, dim=voca_size, name='conv_2'))

    return logit

def get_model(x, voca_size):
    with tf.variable_scope('wavenet'):

        # residual block
        def res_block_keras(x, kernel_size, layer_depth, block, dim=num_dim):
            with tf.name_scope(name='res_block_%d_depth_%d' % (block, layer_depth)):

                dilate_rate = (2 ** layer_depth)
                # filter convolution
                conv_tahn = keras.layers.Conv1D(num_filters,
                                                kernel_size=kernel_size,
                                                dilation_rate= dilate_rate,
                                                activation='tanh',
                                                name='dilated_conv_%d_tahn_s%d' % (dilate_rate, block),
                                                padding='causal')(x)

                # gate convolution
                conv_sigm = keras.layers.Conv1D(num_filters,
                                                kernel_size=kernel_size,
                                                dilation_rate=dilate_rate,
                                                activation='sigmoid',
                                                name='dilated_conv_%d_sigm_s%d' % (dilate_rate, block),
                                                padding='causal')(x)

                # output by gate multiplying

                gated_x = keras.layers.multiply([conv_tahn, conv_sigm],
                                             name='gated_activation_%d_s%d' % (layer_depth, block)
                                             )

                # final output
                res_x = keras.layers.Conv1D(num_filters,
                                       kernel_size=1,
                                       )(gated_x)

                skipped = keras.layers.Conv1D(num_filters,
                                       kernel_size=1,
                                              name='skipped_conv_out'
                                       )(gated_x)
                #res_x = x + res_x
                res_x = keras.layers.add([x, res_x], name='res_add')
                # residual and skip output
                return res_x, skipped

        # expand dimension
        #in_x = keras.layers.Input(shape=(None,feature_size))(x)
        input_conv = keras.layers.Conv1D(num_filters,
                                      kernel_size=1,
                                      dilation_rate=1,
                                      name='initial_causal_conv',
                                      padding='causal')(x)

        # dilated conv block loop
        skipped_conc = [] # skip connections

        for s in range(num_res_blocks):
            for r in range(0, num_res_hidden_layers):
                res_out, skipped = res_block_keras(input_conv, kernel_size=7, layer_depth=r, block=s)
                input_conv = res_out
                skipped_conc.append(skipped)

        # Residual blocks out
        res_out = res_out
        if use_skipped_output:
            res_out = keras.layers.add(skipped_conc)

        res_out_act = keras.layers.Activation('relu')(res_out)

        out_0 = keras.layers.Conv1D(voca_size,
                                      kernel_size = 1,
                                      name = 'out_layer_relu',
                                      activation='relu'
                                      )(res_out)

        out = keras.layers.Conv1D(voca_size,
                                      kernel_size = 1,
                                      name = 'out_layer_softmax',
                                      activation='softmax'
                                      )(out_0)


        return out


