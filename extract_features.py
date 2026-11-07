import io
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# CONFIGURATION
 
DATASET_PATH = Path(
    r"C:\Users\prave\OneDrive\Pictures\Desktop\CASIA2"
)

OUTPUT_DIR = Path("dataset_features")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "image_features.csv"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# Image processing settings
MAX_SIZE = (512, 512)

# LBP settings
P = 8
R = 1
LBP_BINS = P + 2

# IMAGE FEATURE EXTRACTION

def extract_features(image_path):

    with Image.open(image_path) as original:

        image = original.convert("RGB")
        image.thumbnail(MAX_SIZE)

        # ELA FEATURES

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90
        )

        buffer.seek(0)

        with Image.open(buffer) as compressed:
            compressed = compressed.convert("RGB")

            difference = ImageChops.difference(
                image, compressed
            )

            ela = np.asarray(difference, dtype=np.float32)

        # Amplify ELA differences for numerical stability
        ela = np.clip(ela * 10, 0, 255)

        ela_features = [
            float(np.mean(ela)),
            float(np.std(ela)),
            float(np.min(ela)),
            float(np.max(ela)),
            float(np.median(ela)),
            float(np.var(ela))
        ]

        # LBP TEXTURE FEATURES

        gray_image = image.convert("L").resize((256, 256))

        gray = np.asarray(gray_image, dtype=np.uint8)

        lbp = local_binary_pattern(
            gray,
            P,
            R,
            method="uniform"
        )

        histogram, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, LBP_BINS + 1),
            density=False
        )

        histogram = histogram.astype(np.float64)

        histogram /= (histogram.sum() + 1e-10)

        lbp_features = histogram.tolist()

        # IMAGE STATISTICAL FEATURES

        gray_float = gray.astype(np.float64)

        gray_mean = float(np.mean(gray_float))
        gray_std = float(np.std(gray_float))

        # Calculate grayscale entropy
        hist, _ = np.histogram(
            gray,
            bins=256,
            range=(0, 256)
        )

        probabilities = hist / hist.sum()

        probabilities = probabilities[probabilities > 0]

        entropy = float(
            -np.sum(probabilities * np.log2(probabilities))
        )

        rgb = np.asarray(image, dtype=np.float64)

        red_mean = float(np.mean(rgb[:, :, 0]))
        green_mean = float(np.mean(rgb[:, :, 1]))
        blue_mean = float(np.mean(rgb[:, :, 2]))

        image_features = [
            gray_mean,
            gray_std,
            entropy,
            red_mean,
            green_mean,
            blue_mean
        ]

        # COMBINE ALL 22 FEATURES

        features = (
            ela_features
            + lbp_features
            + image_features
        )

        return features


 
# CREATE DATASET FILE LIST
 

image_records = []

class_folders = {
    "Au": 0,   # Genuine
    "Tp": 1    # Edited
}

for folder, label in class_folders.items():

    folder_path = DATASET_PATH / folder

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder_path}"
        )

    for path in folder_path.rglob("*"):

        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:

            image_records.append((path, label))


print("Total images found:", len(image_records))


 
# EXTRACT FEATURES
 

feature_names = [
    "ela_mean",
    "ela_std",
    "ela_min",
    "ela_max",
    "ela_median",
    "ela_variance"
]

feature_names += [
    f"lbp_{i}" for i in range(LBP_BINS)
]

feature_names += [
    "gray_mean",
    "gray_std",
    "gray_entropy",
    "red_mean",
    "green_mean",
    "blue_mean"
]

rows = []
failed_images = []

for index, (image_path, label) in enumerate(
    tqdm(image_records, desc="Extracting image features"),
    start=1
):

    try:

        features = extract_features(image_path)

        if len(features) != 22:
            raise ValueError("Unexpected feature count")

        row = {
            "image_name": image_path.name,
            **dict(zip(feature_names, features)),
            "label": label
        }

        rows.append(row)

    except Exception as error:

        failed_images.append({
            "image": str(image_path),
            "error": str(error)
        })

    # Save intermediate results every 500 images
    if index % 500 == 0:

        pd.DataFrame(rows).to_csv(
            OUTPUT_FILE,
            index=False
        )


 
# SAVE FINAL CSV

df = pd.DataFrame(rows)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# Save failed image information
if failed_images:

    pd.DataFrame(failed_images).to_csv(
        OUTPUT_DIR / "failed_images.csv",
        index=False
    )


# DISPLAY SUMMARY

print("\n========== EXTRACTION SUMMARY ==========")

print("Successfully processed:", len(df))

print("Failed images:", len(failed_images))

print("Number of numerical features:", len(feature_names))

print("\nClass distribution:")

print(df["label"].value_counts().sort_index())

print("\nMissing values:", df[feature_names].isnull().sum().sum())

print("\nOutput saved at:", OUTPUT_FILE)

print("\nFeature extraction completed!")
