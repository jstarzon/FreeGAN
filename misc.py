import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from Neural import model_dir
import os
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

input = 'input'
output = 'output'
model = load_model(os.path.join(model_dir, 'model.h5'))
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
def generator(model):
    # Load the pre-trained generator model

    # Set paths to model and output directory
    output_dir = 'output/'

    # Load GAN model


    # Generate random noise vector
    noise = np.random.normal(1, 0, (322, 100))

    # Generate image from noise vector
    generated_image = model.predict(noise)
    plt.imshow(generated_image[0, :, :, :])

    # Save generated image to output directory
    plt.savefig(os.path.join(output_dir, 'generated_image.png'))

    plt.show()

    
def is_pic_true(model):


    # Load the pre-trained model
    model = load_model('model.h5')

    # Define the image size that the model was trained on
    img_size = (64, 64)

    # Load the image to be checked
    img = cv2.imread(input,'8.jpg')

    # Resize the image to match the model's input size
    img_resized = cv2.resize(img, img_size)

    # Convert the image to a numpy array and normalize the pixel values
    img_array = np.array(img_resized) / 255.0

    # Reshape the image to match the input shape of the model
    img_reshaped = img_array.reshape(1, img_size[0], img_size[1], 3)

    # Make a prediction using the model
    prediction = model.predict(img_reshaped)

    # Check if the prediction is above a certain threshold (e.g. 0.5)
    threshold = 0.5
    if prediction[0][0] > threshold:
        print("The image matches the model!")
    else:
        print("The image does not match the model.")


if __name__ == '__main__':
    
    #generator(model)
    is_pic_true(model)