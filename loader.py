import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt



def pics_loader(pics_path):
    np.random.seed(42)
    tf.set_random_seed(42)

    # Load images
    pics = []

    for filename in os.listdir(pics_path):
        if filename.endswith(".jpg"):
            image = plt.imread(os.path.join(pics_path, filename))
            #grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            pics.append(image)
            print("Loading the pic "+ filename)
    pics = np.array(pics)
    # Normalize images
    pics = (pics - 127.5) / 127.5

    return pics