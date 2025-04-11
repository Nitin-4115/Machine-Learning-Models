import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Ask user to input the image path
image_path = input("Enter the path to the bird image: ").strip()

# Check if image exists
if not os.path.exists(image_path):
    print(f"Image not found at: {image_path}")
    exit()

# Paths
model_path = 'bird_species_model.pth'
train_dir = r"C:\Projects\Bird Species Detection ML Model\dataset\train"

# Load class names
class_names = sorted(os.listdir(train_dir))

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load and transform image
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Prediction
with torch.no_grad():
    input_tensor = input_tensor.to(device)
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]

print(f"Predicted bird species: {predicted_class}")

# Show image with prediction
plt.imshow(mpimg.imread(image_path))
plt.title(f"Predicted: {predicted_class}", fontsize=16)
plt.axis('off')
plt.show()
