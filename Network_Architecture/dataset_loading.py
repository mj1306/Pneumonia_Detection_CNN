#%%Imports
import numpy as np
import tensorflow as tf
import keras as krs
import matplotlib.pyplot as plt

#%%
BATCH_SIZE = 32
IMG_SIZE = (160,160) #Fixed image size in-case dataset consists of images of different sizes
train_directory = r"datasets\Bone_Break_Classification\train"
valid_directory = r"datasets\Bone_Break_Classification\valid"

#%%
train_dataset = krs.preprocessing.image_dataset_from_directory(train_directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             seed=42)
validation_dataset = krs.preprocessing.image_dataset_from_directory(valid_directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             seed=42)

#%%Test to see labelled images
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# %%
