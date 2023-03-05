from models import generator64, make_generator_model
from Neural import date, model_dir, generated, single_pics
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
from telemetry import date

now = date()


def generate_image():
    print("Image generation ...\t", end="")
    generator = generator64()
    generator.load_weights(os.path.join(model_dir + '/generator_weights.h5'))
    noise = np.random.normal(0, 1, (1, 100))
    fake = generator.predict(noise)
    image_name = "image_{}.png".format(now.strftime("%m%d%Y%H%M%S"))
    print(image_name)
    plt.imshow(fake[0, :, :, :])
    plt.savefig(os.path.join(single_pics, image_name))
    

