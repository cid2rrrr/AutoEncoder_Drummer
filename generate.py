import os
import pickle
import librosa
import random

import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from autoencoder import VAE
from train import SPECTROGRAMS_PATH
from preprocess import MinMaxNormaliser

HOP_LENGTH = 470
SAVE_DIR= "./generated/"
MIN_MAX_VALUES_PATH = "./datasets/min_max_values.pkl"


def load_spec(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.dstack(x_train)
    x_train = np.rollaxis(x_train, axis=-1)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model")
    
    
    decoder = vae.decoder
    
    j = decoder.predict(np.random.randn(1,64))
    
    print(j.shape)
    
    
    
    
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    
    d = min_max_values[random.choice(list(min_max_values.keys()))]

    _min_max_normaliser = MinMaxNormaliser(0,1)
    
    signals = []

    log_spectrogram = j.reshape((256,256))

    denorm_log_spec = _min_max_normaliser.denormalise(
        log_spectrogram, d['min'], d['max']) 
        
    # log spectrogram -> spectrogram
    spec = librosa.db_to_amplitude(denorm_log_spec)
    
    # apply Griffin-Lim
    signal = librosa.istft(spec, hop_length=HOP_LENGTH)
    
    # append signal to "signals"
    signals.append(signal)
        

    save_signals(signals, SAVE_DIR)

    




