#%%
import numpy as np
import tensorflow as tf
import keras as krs
import matplotlib.pyplot as plt
import json
from dataset_loading import train_dataset, validation_dataset, test_dataset, data_augment
import os

#%%
resnet50_base = tf.keras.applications.ResNet50(weights = "imagenet", include_top = False, input_shape = (256,256,3))

resnet50_base.trainable = True

fine_tune_at = 140

if resnet50_base.trainable == True:
    for layer in resnet50_base.layers[ :fine_tune_at]:
        layer.trainable = False

model = tf.keras.models.Sequential([
    #data_augment(),
    resnet50_base,
    tf.keras.layers.GlobalAveragePooling2D(),  # GPL
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),  # FC layer with 128 neurons
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

#%%
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-5),
              loss='binary_crossentropy',  
              metrics=['accuracy'])

model.summary()

#%%
history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=10)   

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')


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
file_path = 'ResNet_history.txt'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        try:
            all_history = json.load(f)
        except json.JSONDecodeError:
            all_history = {}
else:
    all_history = {}

next_index = len(all_history) + 1
training_key = f"training ({next_index})"

all_history[training_key] = history.history
#%%
with open(file_path, 'w') as f:
    f.write(json.dumps(history.history, indent=4))