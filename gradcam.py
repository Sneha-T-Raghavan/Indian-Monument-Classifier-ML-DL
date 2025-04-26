import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2

# Load the pre-trained DenseNet model
model = load_model('ml_monument_classifier_model_densenet.h5')

# Check model summary to find the correct last convolutional layer name
model.summary()

# Update this name based on your model summary
last_conv_layer_name = 'densenet121'  # Replace with correct layer name

# Function to generate Grad-CAM heatmap
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    # Check and initialize model input tensors
    if model.input is None:
        # Pass a dummy input to initialize tensors
        dummy_input = tf.random.uniform((1, *img_array.shape[1:]))  # Match img_array shape
        _ = model(dummy_input)  # Forward pass

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Forward pass and gradient computation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    # Weight the output feature maps
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads[..., tf.newaxis], axis=-1)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# Update the correct last convolutional layer name
last_conv_layer_name = "densenet121"  # Replace this with the correct layer name from `model.summary()`


# Preprocessing function for input images
def preprocess_image(img_path, target_size):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array, np.array(img)

# Function to overlay heatmap on the original image
def overlay_heatmap(heatmap, img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # Scale the original image to [0, 255] and cast to uint8
    img = (img * 255).astype('uint8')
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

# Load and preprocess the input image
img_path = "C:/Amrita/Sem 5/ML/CNN/Webscrapped_Data/testing data/City Palace/Image_2.jpg"
target_size = (128, 128)
img_array, original_img = preprocess_image(img_path, target_size)

# Generate the Grad-CAM heatmap
heatmap = generate_gradcam_heatmap(model, img_array, last_conv_layer_name)

# Apply the heatmap to the original image
overlay = overlay_heatmap(heatmap, original_img / 255.0)  # Ensure normalized input

# Plot the results
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(original_img.astype('uint8'))
plt.axis('off')
plt.title('Original Image')

# Grad-CAM heatmap
plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap='viridis')
plt.axis('off')
plt.title('Grad-CAM Heatmap')

# Overlay image
plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.axis('off')
plt.title('Overlay Image')

plt.tight_layout()
plt.show()
