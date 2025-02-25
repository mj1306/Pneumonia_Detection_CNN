#%%Imports
import numpy as np
import tensorflow as tf
import keras as krs
import matplotlib.pyplot as plt

#%%
BATCH_SIZE = 32
IMG_SIZE = (256,256) #Fixed image size in-case dataset consists of images of different sizes
train_directory = r"datasets\chest_xray\chest_xray\train"
valid_directory = r"datasets\chest_xray\chest_xray\val"
test_directory = r"datasets\chest_xray\chest_xray\test"

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
test_dataset = krs.preprocessing.image_dataset_from_directory(test_directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             seed=42)

#for image, label in train_dataset.take(1):  
#    print(image.shape)

#%%Data Augmenter

def data_augment():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

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
