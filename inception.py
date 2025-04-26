from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Rescaling for training data (without augmentation)
train_datagen = ImageDataGenerator(rescale=1.0/255)

# Rescaling for test data
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load train and test datasets
train_data = train_datagen.flow_from_directory(
    r"C:/Amrita/Sem 5/ML/CNN/Augmented Dataset/training data",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    r"C:/Amrita/Sem 5/ML/CNN/Augmented Dataset/testing data",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Ensure no shuffling for evaluation consistency
)

# Load the InceptionV3 base model with pre-trained weights
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Initially freeze all layers

# Unfreeze the top layers for fine-tuning
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Create the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Converts feature maps to a single vector per image
    layers.Dense(256, activation='relu', kernel_regularizer='l2'),  # Fully connected layer with L2 regularization
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(len(train_data.class_indices), activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy', metrics.Precision(), metrics.Recall()]
)

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data,
    callbacks=[lr_scheduler, early_stopping]
)

# Evaluate the model on the test data
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")

# Predict the class probabilities
y_true = test_data.classes  # True labels
class_labels = list(test_data.class_indices.keys())  # Class names
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)  # Predicted class indices

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Generate confusion matrix and plot heatmap
cm = confusion_matrix(y_true, y_pred)

model.save('monument_classifier_model_inception.h5')

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# Plot training and validation metrics
plt.figure(figsize=(12, 5))
# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()



from lime import lime_image
from skimage.segmentation import mark_boundaries

# Wrap model prediction function
def predict_fn(images):
    return model.predict(images)

# Select a sample image
sample_batch = next(test_data)  # Use the built-in `next()` function
sample_images, sample_labels = sample_batch

sample_image = sample_images[0] / 255.0 

# Instantiate LIME explainer
explainer = lime_image.LimeImageExplainer()

# Explain the prediction
explanation = explainer.explain_instance(
    sample_image,
    predict_fn,
    top_labels=5,
    hide_color=0,
    num_samples=1000
)

# Visualize explanation
temp, mask = explanation.get_image_and_mask(
    label=explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=False
)

plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.title("LIME Explanation")
plt.axis("off")
plt.show()

import shap
import numpy as np

# Prepare background data
background_data = train_data.next()[0][:100]  # Use 100 samples for the background

sample_batch = next(test_data)  # Use the built-in `next()` function
sample_images, sample_labels = sample_batch  # Unpack the images and labels

# Select a single image from the batch for LIME
sample_image = sample_images[0] / 255.0

# Instantiate SHAP explainer
explainer = shap.DeepExplainer(model, background_data)

# Compute SHAP values
shap_values = explainer.shap_values(sample_images)

# Visualize SHAP explanations
shap.image_plot(shap_values, sample_images)
