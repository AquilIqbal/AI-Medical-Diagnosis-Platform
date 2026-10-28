from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DATA_FILE = Path(
    r"C:\Users\aquil\AI-Medical-Diagnosis-Platform"
    r"\datasets\image\pneumoniamnist\pneumoniamnist.npz"
)

OUTPUT_DIR = Path("test_xrays")

IMAGES_PER_CLASS = 10


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print("Loading PneumoniaMNIST dataset...")

data = np.load(DATA_FILE)

images = data["test_images"]
labels = data["test_labels"]

print(f"Images shape : {images.shape}")
print(f"Labels shape : {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")


# ---------------------------------------------------------
# PREPARE OUTPUT
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# SHOW CLASS DISTRIBUTION
# ---------------------------------------------------------

print("\nClass distribution:")

for label in np.unique(labels):
    count = int(np.sum(labels == label))
    print(f"  Label {label}: {count} images")


# ---------------------------------------------------------
# EXTRACT IMAGES
# ---------------------------------------------------------

saved_total = 0

for label in np.unique(labels):

    indices = np.where(labels == label)[0]

    selected_indices = indices[:IMAGES_PER_CLASS]

    print(
        f"\nExtracting {len(selected_indices)} "
        f"images from label {label}..."
    )

    for number, index in enumerate(selected_indices, start=1):

        image = np.squeeze(images[index])

        # Convert to uint8.
        image = image.astype(np.float32)

        min_value = image.min()
        max_value = image.max()

        if max_value > min_value:
            image = (
                (image - min_value)
                / (max_value - min_value)
                * 255
            )

        image = image.clip(0, 255).astype(np.uint8)

        output_file = (
            OUTPUT_DIR
            / f"label_{label}_image_{number:02d}"
            f"_index_{index}.png"
        )

        Image.fromarray(image, mode="L").save(output_file)

        print(f"  Saved: {output_file}")

        saved_total += 1


# ---------------------------------------------------------
# DONE
# ---------------------------------------------------------

print("\n==========================================")
print("Extraction complete!")
print("==========================================")
print(f"Total images saved: {saved_total}")
print(f"Output directory: {OUTPUT_DIR.resolve()}")
print("==========================================")