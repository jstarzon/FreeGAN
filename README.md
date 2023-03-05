# FreeGAN
Generate your own GAN models for free

Change this value in training.py
num_epochs = 100
Higher value, better result. This value changes how much iterations training model does through dataset.


Work in progress...

## What it does
- Load pictures dataset
- Train model 
- Saves trained model into .h5 file
- Saves generated pictures into a folder
- Saves telemetry for each epoch in .csv file
- Resume training

## TODO
- Main Menu (Reset model, Resume training, Change dataset, Resolution picker)
- Fancy graphs, realtime image viewer, Iteration counter (epochs), 
- Code cleaning (*μ_μ)
- Training values tweak so model is more accurate


You can put any dataset to train. I decided to use anime pictures.

## Demo (4h model)
![](https://github.com/jstarzon/Anime-GAN-Neural-network/blob/main/evo.gif)

## Old version:
https://github.com/jstarzon/Anime-GAN-Neural-network
