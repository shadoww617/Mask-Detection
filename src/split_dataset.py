import os
import random
import shutil

IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"

TRAIN_IMG = "dataset/images/train"
VAL_IMG = "dataset/images/val"
TRAIN_LBL = "dataset/labels/train"
VAL_LBL = "dataset/labels/val"

os.makedirs(TRAIN_IMG, exist_ok=True)
os.makedirs(VAL_IMG, exist_ok=True)
os.makedirs(TRAIN_LBL, exist_ok=True)
os.makedirs(VAL_LBL, exist_ok=True)

images = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]

random.shuffle(images)

split = int(0.8 * len(images))
train_files = images[:split]
val_files = images[split:]

for f in train_files:
    base = f.rsplit(".", 1)[0]
    shutil.move(os.path.join(IMG_DIR, f), os.path.join(TRAIN_IMG, f))
    shutil.move(os.path.join(LBL_DIR, base + ".txt"), os.path.join(TRAIN_LBL, base + ".txt"))

for f in val_files:
    base = f.rsplit(".", 1)[0]
    shutil.move(os.path.join(IMG_DIR, f), os.path.join(VAL_IMG, f))
    shutil.move(os.path.join(LBL_DIR, base + ".txt"), os.path.join(VAL_LBL, base + ".txt"))

print("Dataset split completed successfully")
