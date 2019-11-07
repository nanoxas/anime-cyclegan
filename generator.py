from keras.layers import *
from keras.models import Model
from keras.utils import multi_gpu_model
from custom_layers import *
from keras_contrib.layers import InstanceNormalization
from keras.initializers import RandomNormal
from spectral_normalization import *
from keras_self_attention import SeqSelfAttention
import keras.regularizers
from keras.layers.advanced_activations import PReLU


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
    c_layer = PReLU()(c_layer)
    c_layer = Conv2D(input_channels, (1, 1), )(c_layer)

    #c_layer = InstanceNormalization(axis=-1)(c_layer)
    c_layer = PReLU()(c_layer)
    c_layer = Conv2D(
        input_channels,
        kernel_size,
        padding='same',
        strides=stride)(c_layer)

    #c_layer = InstanceNormalization(axis=-1)(c_layer)
    c_layer = PReLU()(c_layer)
    c_layer = Conv2D(output_channels, (1, 1),
                     padding='same')(c_layer)

    if input_channels != output_channels or stride != 1:
        input_tensor = Conv2D(output_channels, (1, 1),
                              padding='same', strides=strides)(input_tensor)

    c_layer = Add()([c_layer, input_tensor])
    return c_layer


def up_resblock(x, ch):
    x = residual_block(x, output_channels=ch)
    x = Conv2DTranspose(ch, kernel_size=(3, 3), padding='same', strides=2)(x)
    return x


def decoder(latent_dim=128, gpus=2):
    ch = 1024
    layer_num = 5
    input_tensor = Input((latent_dim, ))
    x = Dense(512, activation='tanh')(input_tensor)
    x = Dense(512, activation='tanh')(x)
    x = Dense(4 * 4 * ch)(x)
    x = Reshape((4, 4, ch))(x)
    x = up_resblock(x, ch)

    for i in range(layer_num // 2):
        x = up_resblock(x, ch // 2)
        ch = ch // 2
    x = Attention(ch)(x)

    for i in range(layer_num // 2):
        x = up_resblock(x, ch // 2)
        ch = ch // 2

    x = residual_block(x, output_channels=ch, kernel_size=(5, 5))
    x = residual_block(x, output_channels=ch, kernel_size=(5, 5))

    x = Conv2D(3, kernel_size=(7, 7), padding='same', activation='tanh')(x)
    with tf.device('/cpu:0'):
        model = Model(
            inputs=input_tensor, outputs=x)
    model = multi_gpu_model(model, gpus=2)
    model.summary()
    return model


def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same',
               kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # concatenate merge channel-wise with input layer
    g = Concatenate()([g, input_layer])
    return g


def resnet_generator(shape, gpus=2):

    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=shape)
    n_resnet = 9
    g = Conv2D(64, (7, 7), padding='same', kernel_initializer=init)(in_image)
    g = Activation('relu')(g)
    g = Conv2D(128, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(g)
    g = Activation('relu')(g)
    g = Conv2D(256, (3, 3), strides=(2, 2),
               padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Attention(256)(g)
    for _ in range(n_resnet):
        g = resnet_block(256, g)
    g = Attention(2560)(g)

    g = Conv2DTranspose(128, (3, 3), strides=(
        2, 2), padding='same', kernel_initializer=init)(g)
    g = Activation('relu')(g)
    g = Attention(128)(g)
    g = resnet_block(128, g)
    g = resnet_block(128, g)
    g = Conv2DTranspose(
        64, (3, 3), strides=(
            2, 2), padding='same', kernel_initializer=init)(g)
    g = Activation('relu')(g)

    g = resnet_block(64, g)
    g = resnet_block(64, g)
    g = resnet_block(64, g)
    g = Conv2D(3, (7, 7), padding='same', kernel_initializer=init)(g)
    out_image = Activation('tanh')(g)
    with tf.device('/cpu:0'):
        model = Model(in_image, out_image)
    model = multi_gpu_model(model, gpus=gpus)
    return model


def residual_Unet(shape, gpus=2):
    nchannels = 64
    ZF = Input(shape)
    xinput = Conv2D(nchannels, (3, 3), padding='same')(ZF)
    xinput = residual_block(xinput, output_channels=nchannels)
    xinput = residual_block(xinput, output_channels=nchannels)
    cd1 = residual_block(xinput, output_channels=nchannels)

    xinput = Conv2D(nchannels, (3, 3), padding='same', strides=2)(cd1)
    xinput = residual_block(xinput, output_channels=nchannels * 2)
    xinput = residual_block(xinput, output_channels=nchannels * 2)
    cd2 = residual_block(xinput, output_channels=nchannels * 2)
    #att_cd2 = Attention(nchannels*2)(cd2)

    xinput = Conv2D(nchannels, (3, 3), padding='same', strides=2)(cd2)
    xinput = residual_block(xinput, output_channels=nchannels * 4)
    xinput = residual_block(xinput, output_channels=nchannels * 4)
    cd3 = residual_block(xinput, output_channels=nchannels * 4)
    att_cd3 = Attention(nchannels * 4)(cd3)

    xinput = Conv2D(nchannels, (3, 3), padding='same', strides=2)(cd3)
    xinput = residual_block(xinput, output_channels=nchannels * 8)
    xinput = residual_block(xinput, output_channels=nchannels * 8)
    xinput = residual_block(xinput, output_channels=nchannels * 8)

    x = residual_block(xinput, output_channels=nchannels * 8)
    x = residual_block(x, output_channels=nchannels * 8)
    x = residual_block(x, output_channels=nchannels * 8)
    x = Conv2DTranspose(
        nchannels * 4,
        kernel_size=(
            3,
            3),
        padding='same',
        strides=2)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Concatenate()([x, att_cd3])

    x = residual_block(x, output_channels=nchannels * 4)
    x = residual_block(x, output_channels=nchannels * 4)
    x = residual_block(x, output_channels=nchannels * 4)
    x = Conv2DTranspose(
        nchannels * 2,
        kernel_size=(
            3,
            3),
        padding='same',
        strides=2)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Concatenate()([x, cd2])
    x = residual_block(x, output_channels=nchannels * 2)
    x = residual_block(x, output_channels=nchannels * 2)
    x = residual_block(x, output_channels=nchannels * 2)
    x = Conv2DTranspose(
        nchannels,
        kernel_size=(
            3,
            3),
        padding='same',
        strides=2)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Concatenate()([x, cd1])
    print(K.int_shape(x))
    x = residual_block(x, output_channels=nchannels)
    x = residual_block(x, output_channels=nchannels)
    x = residual_block(x, output_channels=nchannels)
    output_res = Conv2D(filters=3, kernel_size=(
        1, 1), padding='same')(x)

    print(K.int_shape(output_res))

    with tf.device('/cpu:0'):
        model = Model(
            inputs=ZF, outputs=output_res)
    if gpus > 1:
        model = multi_gpu_model(model, gpus=gpus)
        print("mgpu")
    model.summary()
    return model
