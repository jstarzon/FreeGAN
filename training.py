import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from telemetry import date, data_csv
now = date()
#Generated fake images folder
generated='generated'

def gpu():
    # Check if an AMD GPU is available (i have AMD D: )
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0 and tf.test.is_gpu_available(cuda_only=False):
        # Use the AMD GPU
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        raise Exception("No AMD GPU found")
    # Verify the GPU is being used
    print("Using GPU:", tf.test.gpu_device_name())

#Resume loop
def resume_training(generator, discriminator, gan, model_dir, animecutepics):
    print("Resuming training from checkpoint:", model_dir)
    generator.load_weights(os.path.join(model_dir, 'generator_weights.h5'))
    discriminator.load_weights(os.path.join(model_dir, 'discriminator_weights.h5'))
    gan.load_weights(os.path.join(model_dir, 'gan_weights.h5'))
    training_loop(generator, discriminator, gan, model_dir, animecutepics)

def training_loop(generator, discriminator, gan, model_dir, animecutepics):
    # Train the GAN
    print("Training...")
    num_epochs = 10000
    batch_size = 512
    for epoch in tqdm(range(num_epochs)):
        idx = np.random.randint(0, animecutepics.shape[0], batch_size)
        real_images = animecutepics[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        x = np.concatenate([real_images, fake_images])
        y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        d_loss, d_acc = discriminator.train_on_batch(x, y)
        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        y = np.ones((batch_size, 1))
        g_loss = gan.train_on_batch(noise, y)
        if (epoch + 1) % 10 == 0:
            print("Epoch:", epoch + 1, "Discriminator Loss:", d_loss, "Accuracy:", d_acc, "Generator Loss:", g_loss)
            plt.imshow(fake_images[0, :, :, :])
            plt.savefig(os.path.join(generated, str(epoch)+now.strftime("%m%d%Y%H%M%S")+".png"))
            data_csv(epoch,d_loss,d_acc,g_loss)
            #plt.show()
        if (epoch + 1) % 50 == 0:
            # Save weights
            generator.save_weights(os.path.join(model_dir, 'generator_weights.h5'))
            discriminator.save_weights(os.path.join(model_dir, 'discriminator_weights.h5'))
            gan.save_weights(os.path.join(model_dir, 'gan_weights.h5'))
            