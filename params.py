LEARNING_RATE = 5e-4
BATCH_SIZE = 8
EPOCHS = 400
FRAME_SIZE = 512
HOP_LENGTH = 470
SAMPLE_RATE = 22050
MONO = True

SPECTROGRAMS_PATH = './datasets/spectrograms/'
MIN_MAX_VALUES_PATH = './datasets/'
AUDIO_PATH = './datasets/audio/'
SAVE_DIR = './generated/'

input_shape = (256, 256, 1)
conv_filters = (256, 128, 64, 32, 16)
conv_kernels = (3, 3, 3, 3, 3)
conv_strides = (2, 2, 2, 2, (2,1))
latent_space_dim = 64
