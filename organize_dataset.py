import os
import shutil

# Paths
BASE_DIR = os.getcwd()
CUB_DIR = os.path.join(BASE_DIR, "CUB_200_2011")
IMAGE_DIR = os.path.join(CUB_DIR, "images")
SPLIT_FILE = os.path.join(CUB_DIR, "train_test_split.txt")
IMAGE_LIST_FILE = os.path.join(CUB_DIR, "images.txt")

# Destination
DEST_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DEST_DIR, "train")
TEST_DIR = os.path.join(DEST_DIR, "test")

# Create destination folders
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# Load image mapping and split info
with open(IMAGE_LIST_FILE) as f:
    id_to_name = {int(line.split()[0]): line.split()[1] for line in f.readlines()}

with open(SPLIT_FILE) as f:
    train_test_split = {int(line.split()[0]): int(line.split()[1]) for line in f.readlines()}

# Move images
for img_id, img_name in id_to_name.items():
    is_train = train_test_split[img_id]
    class_name = img_name.split("/")[0]
    
    src_path = os.path.join(IMAGE_DIR, img_name)
    dest_folder = os.path.join(TRAIN_DIR if is_train else TEST_DIR, class_name)
    os.makedirs(dest_folder, exist_ok=True)

    shutil.copy(src_path, os.path.join(dest_folder, os.path.basename(img_name)))

print("✅ Dataset organized successfully!")
