import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.augmentations.transforms import *
from albumentations.augmentations.geometric.transforms import *
from albumentations.augmentations.crops.transforms import *
from albumentations.augmentations.blur.transforms import *
import cv2
import os

# Directory paths

input_dir = r"C:/Amrita/Sem 5/ML/CNN/Webscrapped_Data/training data/Victoria Memorial"
output_dir = r"C:/Amrita/Sem 5/ML/CNN/augmented"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define augmentation pipeline
augmentation_pipeline = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),  # Use HorizontalFlip instead of the deprecated Flip
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.Resize(128, 128),  # Resize images
])

# Apply augmentation and save images
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)

    if img is not None:
        augmented = augmentation_pipeline(image=img)
        augmented_img = augmented['image']

        # Save augmented image
        save_path = os.path.join(output_dir, f"aug_{img_name}")
        cv2.imwrite(save_path, augmented_img)