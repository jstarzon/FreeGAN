import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Generator model

def generator64():
    print("Generator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128 * 16 * 16, activation="relu", input_shape=(100, )))
    model.add(tf.keras.layers.Reshape((16, 16, 128)))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2D(3, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    return model

def discriminator64():
    print("Discriminator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((64 * 64 * 3,), input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def make_generator_model():
    print("Generator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128 * 16 * 16, activation="relu", input_shape=(100, )))
    model.add(tf.keras.layers.Reshape((16, 16, 128)))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.UpSampling2D())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Conv2D(3, kernel_size=3, padding="same"))
    model.add(tf.keras.layers.Activation("tanh"))
    return model

# Discriminator model
def make_discriminator_model():
    print("Discriminator model in progress")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((64 * 64 * 3,), input_shape=(64, 64, 3)))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def define_gan(discriminator, generator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
