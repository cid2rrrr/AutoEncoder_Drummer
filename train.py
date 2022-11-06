import os

import numpy as np
import tensorflow as tf
from autoencoder import VAE

import params



def load_spec(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) 
            if spectrogram.shape[1] == 256:
                x_train.append(spectrogram)

    x_train = np.dstack(x_train)
    x_train = np.rollaxis(x_train, axis=-1)
    x_train = x_train[..., np.newaxis]
    
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=params.input_shape,
        conv_filters=params.conv_filters,
        conv_kernels=params.conv_kernels,
        conv_strides=params.conv_strides,
        latent_space_dim=params.latent_space_dim
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_spec(params.SPECTROGRAMS_PATH)
    
    with tf.device('/gpu:0'):
        autoencoder = train(x_train, params.LEARNING_RATE, params.BATCH_SIZE, params.EPOCHS)
    autoencoder.save("model")
