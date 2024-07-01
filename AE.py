import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Cropping2D
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import *

# Configure GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Set random seed for reproducibility
np.random.seed(33)

# Constants
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 1
EPOCHS = 100
BATCH_SIZE = 1

def load_data(directory):
    """
    Load images from directory without using ImageDataGenerator.
    Convert RGB images to grayscale.
    :param directory: Directory containing the images.
    :return: Numpy array of grayscale images.
    """
    images = []
    filenames = os.listdir(directory)
    for filename in filenames:
        if filename.endswith(".jpg"):
            img = load_img(os.path.join(directory, filename), target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = img.convert('L')  # Convert to grayscale
            img = img_to_array(img) / 255.0  # normalize to [0, 1]
            images.append(img)
    return np.array(images)

def build_unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)

    # Downsampling
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottleneck
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    # Upsampling
    conv5_up = UpSampling2D(size=(2, 2))(conv5)
    P4 = Concatenate(axis=3)([conv4, conv5_up])
    P4 = Conv2D(512, (3, 3), activation='relu', padding='same')(P4)
    P4 = Conv2D(512, (3, 3), activation='relu', padding='same')(P4)

    conv4_up = UpSampling2D(size=(2, 2))(P4)
    P3 = Concatenate(axis=3)([conv3, conv4_up])
    P3 = Conv2D(256, (3, 3), activation='relu', padding='same')(P3)
    P3 = Conv2D(256, (3, 3), activation='relu', padding='same')(P3)

    conv3_up = UpSampling2D(size=(2, 2))(P3)
    P2 = Concatenate(axis=3)([conv2, conv3_up])
    P2 = Conv2D(128, (3, 3), activation='relu', padding='same')(P2)
    P2 = Conv2D(128, (3, 3), activation='relu', padding='same')(P2)

    conv2_up = UpSampling2D(size=(2, 2))(P2)
    P1 = Concatenate(axis=3)([conv1, conv2_up])
    P1 = Conv2D(64, (3, 3), activation='relu', padding='same')(P1)
    P1 = Conv2D(64, (3, 3), activation='relu', padding='same')(P1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(P1)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def plot_representation(encode_images):
    """
    Plot the hidden result.
    :param encode_images: the images after encoding
    :return:
    """
    plt.scatter(encode_images[:, 0], encode_images[:, 1], s=3)
    plt.colorbar()
    plt.show()

def show_images(decode_images, x_test):
    """
    Plot the images.
    :param decode_images: the images after decoding
    :param x_test: testing data
    :return:
    """
    n = len(decode_images)
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        ax.imshow(x_test[i].reshape(512, 512), cmap='gray')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        ax.imshow(decode_images[i].reshape(512, 512), cmap='gray')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    # Load data
    x_train = load_data('./dataset/train/')
    x_train = x_train.reshape((x_train.shape[0], 512, 512, 1))

    # Build U-Net model
    model = build_unet(input_size=(512, 512, 1))

    # Train the model
    model.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)

    # Test and plot (using the same data for simplicity)
    encode_images = model.predict(x_train)
    plot_representation(encode_images.reshape((-1, 512 * 512)))

    decode_images = model.predict(x_train)
    show_images(decode_images, x_train)