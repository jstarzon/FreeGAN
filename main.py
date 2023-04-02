import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from training import training_loop, resume_training
from loader import pics_loader,model_loader,recent_models
from models import make_generator_model, make_discriminator_model, define_gan
from telemetry import date, data_csv
now = date()
model_dir = 'models'
single_pics='single'
pics_path = 'data/64x64'


def starting(model_gan,model_disc,model_gen):
    #Models folder
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
        print("No models were found")
        training_loop(generator, discriminator, gan, model_dir , pics)
    else:
        resume_training(generator, discriminator, gan, model_dir, pics, model_gan,model_disc,model_gen)
        

def main():

    while True:
        print("Welcome to the GAN Menu!")
        print("1. Start GAN")
        print("2. Stop GAN")
        print("3. Resume GAN")
        print("4. Load Model")
        print("5. Load Weights")
        print("6. Generate Picture")
        print("0. Exit")
        
        choice = input("Enter your choice: ")
        is_model_loaded = False
        if choice == '1':
            print("Starting GAN...")
            if is_model_loaded == True:
                model_gan = models_tab[0]
                model_disc = models_tab[1]
                model_gen = models_tab[2]
                starting(model_gan,model_disc,model_gen)
                
            else:      
                models_tab = model_loader()
                model_gan = models_tab[0]
                model_disc = models_tab[1]
                model_gen = models_tab[2]
                is_model_loaded = True
                starting(model_gan,model_disc,model_gen)
        elif choice == '2':
            print("Stopping GAN...")
        elif choice == '3':
            print("Resuming GAN...")
            models_tab = recent_models(model_dir)
            model_disc = models_tab[1]
            model_gen = models_tab[2]
            model_gan = models_tab[3]
            print(model_gan,model_disc,model_gen)
            starting(model_gan,model_disc,model_gen)
        elif choice == '4':
            is_model_loaded = True
            models_tab = model_loader()
            model_gan = models_tab[0]
            model_disc = models_tab[1]
            model_gen = models_tab[2]
        elif choice == '5':
            print("Loading Weights...")
        elif choice == '6':
            print("Generating Picture...")
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

        
if __name__ == '__main__':
    main()
    