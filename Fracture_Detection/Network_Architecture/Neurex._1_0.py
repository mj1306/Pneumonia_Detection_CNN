import numpy as np
import tensorflow as tf
import keras as krs
import matplotlib.pyplot as plt
import json
from dataset_loading import train_dataset, validation_dataset

vgg_base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
vgg_base.trainable = False
model = tf.keras.models.Sequential([
    vgg_base,  # VGG16 base model without the top layer
    tf.keras.layers.GlobalAveragePooling2D(),  # GPL
    tf.keras.layers.Dense(128, activation='relu'),  # FC layer with 128 neurons
    tf.keras.layers.Dense(11, activation='softmax')  # Softmax output layer for 11 classes 
])

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), 
              loss='sparse_categorical_crossentropy',  # Data labels are integers (not one-hot encoded)
              metrics=['accuracy'])

#model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('model_checkpoint.weights.h5', 
                                      save_best_only=True,  
                                      save_weights_only=True,  
                                      verbose=1)
history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=20,
                    callbacks=[checkpoint])  

model.save('final_model.keras')  

#%% Plot the accuracy and loss over epochs
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')

plt.show()





