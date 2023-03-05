import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from training import training_loop, resume_training
from loader import pics_loader
from models import make_generator_model, make_discriminator_model, define_gan
from telemetry import date

now = date()

#Models folder
model_dir = 'models'
single_pics='single'
pics_path = 'data/64x64'


def main():
    pics = pics_loader(pics_path)
    print("Generator compile DONE")
    generator = make_generator_model()
    print("Discriminator compile DONE")
    discriminator = make_discriminator_model()
    print("Discriminator model compiling in progress...")
    print("Combining.. GAN + Discriminator")
    gan = define_gan(discriminator,generator)
    print("Compiling in progress...")
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        training_loop(generator, discriminator, gan, model_dir , pics)
    else:
        resume_training(generator, discriminator, gan, model_dir, pics)
        


        
if __name__ == '__main__':
    main()