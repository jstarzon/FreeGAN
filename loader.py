import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

def recent_models(folder_path):
    keywords = ["discriminator_weights", "gan_weights", "generator_weights"]
    files = os.listdir(folder_path)
    filtered_files = [f for f in files if any(kw in f for kw in keywords)]
    filtered_files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
    for f in filtered_files[:3]:
        print(os.path.join(folder_path, f))
        return filtered_files

def chose_file(folder_path):
    files = os.listdir(folder_path)

    print("Files in the folder:")
    for i, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"{i+1}. {file_name} (last modified: {modification_time})")

    while True:
        try:
            choice = int(input("Enter the number of the file you want to select: "))
            if choice < 1 or choice > len(files):
                print("Invalid choice. Please try again.")
            else:
                selected_file = os.path.join(folder_path, files[choice-1])
                modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(selected_file))
                print(f"You selected '{selected_file}' (last modified: {modification_time}).")
                return selected_file
        except ValueError:
            print("Invalid input. Please enter a number.")
            
def model_loader():
    print("Loading Model...")
    print("----------------")
    print("Choose GAN model")
    model_gan = chose_file("models")

    print("----------------")
    print("Choose DISCRIMINATOR model")
    model_disc = chose_file("models")

    print("----------------")
    print("Choose GENERATOR model")
    model_gen = chose_file("models")

    print("----------------")
    return model_gan,model_disc,model_gen

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