import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from Neural import model_dir
import os
import cv2
from tensorflow.keras.layers import Flatten

model = load_model(os.path.join(model_dir, '03202023212343model.h5'))
def generator(model):
    # Load the pre-trained generator model

    # Set paths to model and output directory
    output_dir = 'output/'

    # Load GAN model


    # Generate random noise vector
    noise = np.random.normal(1, 1, (1, 100))

    # Generate image from noise vector
    generated_image = model.predict(noise)
    plt.imshow(generated_image[0, :, :, :])

    # Save generated image to output directory
    plt.savefig(os.path.join(output_dir, 'generated_image.png'))

    plt.show()


def is_pic_true(model):

    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    # Load the image and preprocess it
    img = cv2.imread('8.jpg')
    img_tab = []
    img_tab.append(img)
    img = cv2.resize(img,(64,64))
    
    img = np.reshape(img,[1,64,64,3])
    # Make a prediction using the model
    prediction = model.predict_classes(img)
    # Return True if prediction is greater than or equal to 0.5, False otherwise
    return prediction[0, 0] >= 0.5





if __name__ == '__main__':
    generator(model)
    #is_pic_true(model)