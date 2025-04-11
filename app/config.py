import os
from torchvision import transforms

# Paths
BASE_DIR = r"C:\Projects\Bird Species Detection ML Model"
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "bird_species_model.pth")

# Model settings
IMAGE_SIZE = (224, 224)

# Common image transform
TRANSFORM = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])
