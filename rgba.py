import os
from PIL import Image
import numpy as np

# Define directories
train_dir = r"C:/Amrita/Sem 5/ML/CNN/Augmented Dataset/training data"
test_dir = r"C:/Amrita/Sem 5/ML/CNN/Augmented Dataset/testing data"


# Function to fix images with transparency
def fix_transparency(image_path, save_path):
    with Image.open(image_path) as img:
        # Check if image has transparency
        if img.mode == 'RGBA':
            # Convert RGBA to RGB (removes transparency)
            img = img.convert('RGB')
        elif img.mode == 'P':  # Handle palette-based images
            img = img.convert('RGB')
        
        # Save the fixed image, retaining the original file format
        img.save(save_path, format=img.format)

# Iterate through the training data
for subdir, dirs, files in os.walk(train_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Check image extensions
            image_path = os.path.join(subdir, file)
            save_path = image_path  # Overwrite the original
            fix_transparency(image_path, save_path)

# Iterate through the testing data
for subdir, dirs, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Check image extensions
            image_path = os.path.join(subdir, file)
            save_path = image_path  # Overwrite the original
            fix_transparency(image_path, save_path)

print("Transparency issue fixed for all images in training and testing data.")