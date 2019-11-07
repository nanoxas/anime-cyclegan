from keras.layers import *
from keras.models import Model, Sequential
from keras.utils import multi_gpu_model
from custom_layers import Attention
from spectral_normalization import *
from keras.regularizers import l2
import tensorflow as tf
from keras.layers.pooling import _GlobalPooling2D
from keras.initializers import RandomNormal
from keras.applications import VGG16
from keras.layers.core import Activation
from keras_self_attention import SeqSelfAttention
import keras.regularizers
from keras_contrib.layers import InstanceNormalization


class GlobalSumPooling2D(_GlobalPooling2D):
    """Global sum pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        if self.data_format == 'channels_last':
            return K.sum(inputs, axis=[1, 2])
        else:
            return K.sum(inputs, axis=[2, 3])


def residual_block(input_tensor, input_channels=None,
                   output_channels=None, kernel_size=(3, 3), stride=1):
    """
    full pre-activation residual block
    https://arxiv.org/pdf/1603.05027.pdf
    """
    if output_channels is None:
        output_channels = input_tensor.get_shape()[-1].value
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    c_layer = input_tensor
    c_layer = LeakyReLU(alpha=0.2)(c_layer)
    c_layer = Conv2D(input_channels, (1, 1), )(c_layer)
    c_layer = LeakyReLU(alpha=0.2)(c_layer)
    c_layer = Conv2D(
        input_channels,
        kernel_size,
        padding='same',
        strides=stride)(c_layer)

    c_layer = LeakyReLU(alpha=0.2)(c_layer)
    c_layer = Conv2D(output_channels, (1, 1),
                     padding='same')(c_layer)

    if input_channels != output_channels or stride != 1:
        input_tensor = Conv2D(output_channels, (1, 1),
                              padding='same', strides=strides)(input_tensor)

    c_layer = Add()([c_layer, input_tensor])
    return c_layer


def patch_discriminator(shape, gpus=2):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=shape)
    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same',
               kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Attention(256)(d)
    d = Conv2D(512, (4, 4), strides=(2, 2),
               padding='same', kernel_initializer=init)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    x = LeakyReLU(alpha=0.2)(d)
    x = Attention(512)(d)
    output = Conv2D(1, (4, 4), padding='same',
                    activation='sigmoid', kernel_initializer=init)(d)
    with tf.device('/cpu:0'):
        model = Model(in_image, output)
    model = multi_gpu_model(model, gpus=2)

    return model


def build_discriminator_att(shape, gpus=2):
    ch = 64
    layer_num = 5
    input_tensor = Input(shape)
    x = Conv2D(ch, (3, 3), strides=(2, 2),
               padding='same')(input_tensor)  # 112x112
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = residual_block(x, output_channels=ch)
    x = residual_block(x, output_channels=ch * 2, stride=2)
    x = Attention(ch * 2)(x)

    ch = ch * 2

    for i in range(layer_num):
        if i == layer_num - 1:
            x = residual_block(x, output_channels=ch * 2)
        else:
            x = residual_block(x, output_channels=ch * 2, stride=2)

        ch = ch * 2

    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output)

    return model


def build_discriminator_sn(shape, gpus=2):
    model = Sequential()

    model.add(
        Conv2D(
            64, (3, 3), strides=(
                2, 2), padding="same", input_shape=shape))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(512, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(0.2))

    #model.add(Conv2D(1, (3, 3), padding="same"))
    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    return model


def build_discriminator_vgg(shape, gpus=2):
    nchannels = 64

    ZF = Input(shape)
    discriminator = VGG16(include_top=False, input_tensor=ZF)
    print(K.int_shape(discriminator.outputs[0]))
    pool = GlobalAveragePooling2D()(discriminator.outputs[0])

    output = Dense(1, activation='sigmoid')(pool)

    model = Model(
        inputs=ZF, outputs=output)

    # model.summary()
    return model


def layer_abs_sum(x):
    return K.expand_dims(K.mean(K.abs(x), axis=[1, 2, 3]), axis=-1)


def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False
    # discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
    # identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
    # forward cycle
    output_f = g_model_2(gen1_out)
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    # define model graph
    with tf.device('/cpu:0'):
        model = Model([input_gen, input_id], [
                      output_d, output_id, output_f, output_b])
    model = multi_gpu_model(model, gpus=2)
    return model


def add_discriminator_to_generator(g, d):
    d.trainable = False
    input_tensor = Input((128,))
    g_out = g(input_tensor)
    d_on_g = d(g_out)

    model = Model(inputs=input_tensor, outputs=d_on_g)

    return model
