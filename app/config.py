import os
from torchvision import transforms

# Base directory (relative to current file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths (relative to repo structure)
DATASET_PATH = os.path.join(BASE_DIR, "..", "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "..", "bird_species_model.pth")

# Model settings
IMAGE_SIZE = (224, 224)

# Common image transform
TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])
