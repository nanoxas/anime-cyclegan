from keras import backend as K
from keras.layers import Layer, InputSpec
import tensorflow as tf


class Attention(Layer):
    def __init__(self, channels, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = channels
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(
            name='gamma',
            shape=[1],
            initializer='zeros',
            trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,),
                                      initializer='zeros',
                                      name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,),
                                      initializer='zeros',
                                      name='bias_h')
        super(Attention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x_input):
        def hw_flatten(x_input):
            output_shape = [K.shape(x_input)[0], K.shape(
                x_input)[1] * K.shape(x_input)[2], K.shape(x_input)[-1]]

            flattened_input = K.reshape(x_input, shape=output_shape)
            return flattened_input

        f_output = K.conv2d(x_input,
                            kernel=self.kernel_f,
                            strides=(1, 1), padding='same')  # [bs, h_output, w, c']
        f_output = K.bias_add(f_output, self.bias_f)
        g_output = K.conv2d(x_input,
                            kernel=self.kernel_g,
                            strides=(1, 1), padding='same')  # [bs, h_output, w, c']
        g_output = K.bias_add(g_output, self.bias_g)
        h_output = K.conv2d(x_input,
                            kernel=self.kernel_h,
                            strides=(1, 1), padding='same')  # [bs, h_output, w, c]
        h_output = K.bias_add(h_output, self.bias_h)

        s_output = tf.matmul(
            hw_flatten(g_output),
            hw_flatten(f_output),
            transpose_b=True)  # # [bs, N, N]

        beta = K.softmax(s_output, axis=-1)  # attention map

        output_attention = tf.matmul(beta, hw_flatten(h_output))  # [bs, N, C]

        output_attention = K.reshape(
            output_attention,
            shape=K.shape(x_input))  # [bs, h_output, w, C]
        x_input = self.gamma * output_attention + x_input

        return x_input

    def compute_output_shape(self, input_shape):
        return input_shape
