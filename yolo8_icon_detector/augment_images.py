import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Configuration
INPUT_ROOT = "yolo8_icon_detector/data"
AUGMENTED_COUNT = 50  # Number of augmented versions per original image
OUTPUT_SUFFIX = "_aug"

# Define augmentation pipeline
augmentation_pipeline = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

def augment_and_save(image_path, output_dir, base_name, count):
    original = Image.open(image_path).convert("RGB")

    for i in range(count):
        aug_img = augmentation_pipeline(original)
        output_path = os.path.join(output_dir, f"{base_name}{OUTPUT_SUFFIX}_{i}.png")
        aug_img.save(output_path)

def run_augmentation():
    for class_name in os.listdir(INPUT_ROOT):
        class_dir = os.path.join(INPUT_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(class_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            augment_and_save(img_path, class_dir, base_name, AUGMENTED_COUNT)

    print("✅ Augmentation complete! Augmented images saved next to originals.")

if __name__ == "__main__":
    run_augmentation()