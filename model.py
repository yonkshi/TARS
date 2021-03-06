import tensorflow as tf
import keras
num_dilation = 3     # dilated blocks
num_dim = 128      # latent dimension
num_filters = 5
use_skipped_output = True # flag to choose if use resnet
num_res_blocks = 3  # Number of res_net blocks
num_res_hidden_layers = 3 # Number of res net hidden layers
feature_size = 13 # Input dimension

def build_wavenet(x, voca_size):
    '''
    MAin building block for WaveNet
    :param x: input tensor
    :param voca_size: output dimension
    :return:
    '''
    with tf.variable_scope('wavenet'):

        # residual block
        def res_block_keras(x, kernel_size, layer_depth, block, dim=num_dim):
            '''
            Residual block builder, uses a special residual block with gated convolution as specificed in WaveNet paper
            :param x: input tensor
            :param kernel_size: kernel size
            :param layer_depth: Depth in residual block, only used in naming components
            :param block: Block number in residual blocks, only used in naming components
            :param dim:
            :return:
            '''
            with tf.name_scope(name='res_block_%d_depth_%d' % (block, layer_depth)):

                dilate_rate = (2 ** layer_depth)
                # filter convolution
                conv_tahn = keras.layers.Conv1D(num_filters,
                                                kernel_size=kernel_size,
                                                dilation_rate= dilate_rate,
                                                activation='tanh',
                                                name='dilated_conv_%d_tahn_s%d' % (dilate_rate, block),
                                                padding='causal')(x)
                #conv_tahn = keras.layers.BatchNormalization()(conv_tahn)

                # gate convolution
                conv_sigm = keras.layers.Conv1D(num_filters,
                                                kernel_size=kernel_size,
                                                dilation_rate=dilate_rate,
                                                activation='sigmoid',
                                                name='dilated_conv_%d_sigm_s%d' % (dilate_rate, block),
                                                padding='causal')(x)
                #conv_sigm = keras.layers.BatchNormalization()(conv_sigm)

                # output by gate multiplying

                gated_x = keras.layers.multiply([conv_tahn, conv_sigm],
                                             name='gated_activation_%d_s%d' % (layer_depth, block)
                                             )

                # final output
                res_x = keras.layers.Conv1D(num_filters,
                                       kernel_size=1,
                                       )(gated_x)
                #res_x = keras.layers.BatchNormalization()(res_x)

                skipped = keras.layers.Conv1D(num_filters,
                                       kernel_size=1,
                                              name='skipped_conv_out'
                                       )(gated_x)
                #res_x = x + res_x
                res_x_added = keras.layers.add([x, res_x], name='res_add')
                # residual and skip output
                return res_x_added, res_x

        # expand dimension
        #x = keras.layers.BatchNormalization()(x)
        input_conv = keras.layers.Conv1D(num_filters,
                                      kernel_size=1,
                                      dilation_rate=1,
                                      name='initial_causal_conv',
                                      padding='causal')(x)
        #input_conv = keras.layers.BatchNormalization()(input_conv)

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
                                      name = 'before_relu',
                                      activation='relu'
                                      )(res_out)
        #out_0 = keras.layers.BatchNormalization()(out_0)

        out = keras.layers.Conv1D(voca_size,
                                      kernel_size = 1,
                                      name = 'output_layer',
                                      #activation='softmax'
                                      )(out_0)
        softmax_out = tf.nn.softmax(out)


        return softmax_out, out


