import os
from torchvision import transforms
from config import IMAGE_SIZE

def load_class_names(_=None):
    class_file_path = os.path.join(os.path.dirname(__file__), "..", "class_names.txt")
    with open(class_file_path, "r") as f:
        return [line.strip() for line in f]

def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
