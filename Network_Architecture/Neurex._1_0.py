#%%
import numpy as np
import tensorflow as tf
import keras as krs
import matplotlib.pyplot as plt
import json
from dataset_loading import train_dataset, validation_dataset, data_augment

#%%
resnet50_base = tf.keras.applications.ResNet50(weights = "imagenet", include_top = False, input_shape = (256,256,3))

resnet50_base.trainable = True

fine_tune_at = 140

for layer in resnet50_base.layers[ :fine_tune_at]:
    layer.trainable = False

model = tf.keras.models.Sequential([
    data_augment(),
    resnet50_base,
    tf.keras.layers.GlobalAveragePooling2D(),  # GPL
    tf.keras.layers.Dense(128, activation='relu'),  # FC layer with 128 neurons
    #tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(11, activation='softmax')  # Softmax output layer for 11 classes 
])

#%%
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), 
              loss='sparse_categorical_crossentropy',  # Data labels are integers (not one-hot encoded)
              metrics=['accuracy'])

model.summary()

#%%
history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=10)   

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

#%%
with open('training_history.txt', 'w') as f:
    f.write(json.dumps(history.history, indent=4))
