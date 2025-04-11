import os
from torchvision import transforms
from config import IMAGE_SIZE

def load_class_names(dataset_path):
    return sorted(os.listdir(dataset_path))

def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
