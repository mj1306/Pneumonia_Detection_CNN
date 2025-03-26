from tensorflow import keras
from dataset_loading import train_dataset, validation_dataset, test_dataset, data_augment

# Load the full model
model = keras.models.load_model("vgg_model_full.h5")

# Recompile the model to restore metrics
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Now you can evaluate or predict
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy}")
