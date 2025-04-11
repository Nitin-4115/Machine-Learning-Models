import os
from torchvision import transforms
from config import IMAGE_SIZE

# Resolve path to class_names.txt relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "class_names.txt")

def load_class_names(_=None):
    with open(CLASS_NAMES_PATH, "r") as f:
        return [line.strip() for line in f]

def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
