import numpy as np
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from generator import *
from discriminator import *
from dataset_builder import *
import matplotlib.pyplot as plt
from keras.callbacks import *


def generate_real_samples(dataset, n_samples, patch_size):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, patch_size, patch_size, 1))
    return X, y


def generate_fake_samples(g_model, dataset, patch_size):
    X = g_model.predict(dataset)
    y = np.zeros((len(X), patch_size, patch_size, 1))
    return X, y


def to_anime_file(gen_path, photo_path):
    img_orig = np.array(Image.open(photo_path))
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGBA2RGB)
    img_orig = (img_orig / 255) * 2 - 1
    print(img_orig.shape)

    generator_A2B = resnet_generator(img_orig.shape)
    generator_A2B.load_weights(gen_path)
    predicted = generator_A2B.predict(np.expand_dims(img_orig, axis=0))
    im = np.uint8(predicted[0, ...] * 127.5 + 127.5)
    orig = np.uint8(img_orig * 127.5 + 127.5)
    im_c = np.concatenate((im, orig), axis=1)
    plt.imsave(photo_path + '_conv.png', im_c)


def to_anime(gen_path, dataset):

    generator_A2B = resnet_generator((64, 64, 3))
    generator_A2B.load_weights(gen_path)
    predicted = generator_A2B.predict(dataset)
    for i in range(len(predicted)):
        im = np.uint8(predicted[i, ...] * 127.5 + 127.5)
        orig = np.uint8(dataset[i, ...] * 127.5 + 127.5)
        im_c = np.concatenate((im, orig), axis=1)
        plt.imsave('./to_anime/human_anime_conv' + str(i) + '.png', im_c)


def generate_sample(dec, idx):
    seed = np.random.normal(0, 1, (1, 128))
    anime_face = dec.predict(seed)

    fig = plt.figure(dpi=700)
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(np.uint8(anime_face[0, ...] * 127.5 + 127.5))
    fig.savefig('gen_anime' + str(idx) + '.png')
    plt.close(fig)


def sample_images(generatorS2T, source, translation_name, idx):

    predicted = generatorS2T.predict(source)
    im = np.uint8(predicted[0, ...] * 127.5 + 127.5)
    im_source = np.uint8(source[0, ...] * 127.5 + 127.5)
    im_c = np.concatenate((im, im_source), axis=1)
    plt.imsave('./outputs/' + translation_name + str(idx) + '.png', im_c)


def train_cycgan(ATrain, BTrain, shape):
    generator_A2B = resnet_generator(shape)
    generator_B2A = resnet_generator(shape)

    discriminator_A = patch_discriminator(shape)
    discriminator_B = patch_discriminator(shape)
    optd = Adam(0.0001, 0.5)
    optg = Adam(0.0001, 0.5)
    discriminator_B.compile(loss='binary_crossentropy', optimizer=optd)
    discriminator_A.compile(loss='binary_crossentropy', optimizer=optd)

    c_model_AtoB = define_composite_model(
        generator_A2B, discriminator_B, generator_B2A, shape)
    c_model_AtoB.compile(
        loss=[
            'binary_crossentropy', 'mae', 'mae', 'mae'], loss_weights=[
            1, 5, 10, 10], optimizer=optg)
    c_model_BtoA = define_composite_model(
        generator_B2A, discriminator_A, generator_A2B, shape)
    c_model_BtoA.compile(
        loss=[
            'binary_crossentropy', 'mae', 'mae', 'mae'], loss_weights=[
            1, 5, 10, 10], optimizer=optg)

    n_epochs, n_batch, = 100, 16
    bat_per_epo = int(len(ATrain) / n_batch)
    patch_size = 4
    n_steps = bat_per_epo * n_epochs

    avg_losses = [0] * 6
    for i in range(n_steps):
        X_realA, y_realA = generate_real_samples(
            ATrain, n_batch, patch_size)
        X_realB, y_realB = generate_real_samples(
            BTrain, n_batch, patch_size)
        X_fakeA, y_fakeA = generate_fake_samples(
            generator_B2A, X_realB, patch_size)
        X_fakeB, y_fakeB = generate_fake_samples(
            generator_A2B, X_realA, patch_size)

        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch(
            [X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

        dA_loss1 = discriminator_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = discriminator_A.train_on_batch(X_fakeA, y_fakeA)

        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch(
            [X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        dB_loss1 = discriminator_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = discriminator_B.train_on_batch(X_fakeB, y_fakeB)

        avg_losses[0] = avg_losses[0] + \
            (1 / (i + 1)) * (dA_loss1 - avg_losses[0])
        avg_losses[1] = avg_losses[1] + \
            (1 / (i + 1)) * (dA_loss2 - avg_losses[1])
        avg_losses[2] = avg_losses[2] + \
            (1 / (i + 1)) * (dB_loss1 - avg_losses[2])
        avg_losses[3] = avg_losses[3] + \
            (1 / (i + 1)) * (dB_loss2 - avg_losses[3])

        avg_losses[4] = avg_losses[4] + \
            (1 / (i + 1)) * (g_loss1 - avg_losses[4])
        avg_losses[5] = avg_losses[5] + \
            (1 / (i + 1)) * (g_loss2 - avg_losses[5])

        if i % 100 == 0:
            sample_images(
                generator_A2B, X_realA[0:1, ...], 'human_anime', i)
            sample_images(
                generator_B2A, X_realB[0:1, ...], 'anime_human', i)
        print(
            '>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' %
            (i + 1, avg_losses[0], avg_losses[1], avg_losses[2], avg_losses[3], avg_losses[4], avg_losses[5]))
        if i % 500 == 0 and i > 0:
            generator_A2B.save_weights('human_anime_generator', True)
            generator_B2A.save_weights('anime_human_generator', True)
            discriminator_A.save_weights('discriminator_human', True)
            discriminator_B.save_weights('discriminator_anime', True)


def train_gan(BTrain):
    input_shape = (BTrain.shape[1], BTrain.shape[2], BTrain.shape[3])
    anime_discriminator = patch_discriminator(input_shape)
    anime_generator = decoder()
    opt = Adam(0.0001, 0.5)
    anime_discriminator.compile(
        loss='binary_crossentropy',
        optimizer=opt)
    composite_model = add_discriminator_to_generator(
        anime_generator, anime_discriminator)
    composite_model.compile(loss='binary_crossentropy', optimizer=opt)

    n_epochs, n_batch, = 100, 64
    bat_per_epo = int(len(BTrain) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        anime_discriminator.trainable = True
        for j in range(1):
            X_realB, y_realB = generate_real_samples(BTrain, n_batch)
            X_fakeB, y_fakeB = generate_fake_samples(
                anime_generator, np.random.normal(0, 1, (n_batch, 128)))

            d_loss_real = anime_discriminator.train_on_batch(
                X_realB, y_realB)
            d_loss_fake = anime_discriminator.train_on_batch(
                X_fakeB, y_fakeB)
            d_loss = (d_loss_real + d_loss_fake) / 2

        noise = np.random.normal(0, 1, size=(n_batch, 128))
        anime_discriminator.trainable = False
        g_loss = composite_model.train_on_batch(noise, y_realB)
        #pre_g_loss = anime_generator.train_on_batch(noise, X_realB)
        #d_loss = (d_loss_true + d_loss_fake) / 2
        print(
            "%d [D loss: %f] [G loss: %f]" %
            (i, d_loss, g_loss))
        if i % 100 == 0:
            generate_sample(anime_generator, i)
            anime_generator.save_weights('anime_generator', True)
            anime_discriminator.save_weights('anime_discriminator', True)


if __name__ == "__main__":
    ATrain, BTrain = read_faces(
        r'/media/gabriel/TOSHIBA EXT/img_align_celeba/', r'/home/gabriel/anime-faces/')
    print(ATrain.shape)
    print(BTrain.shape)
    train_cycgan(
        ATrain,
        BTrain,
        (ATrain.shape[1],
         ATrain.shape[2],
         ATrain.shape[3]))

    # train_gan(BTrain)
