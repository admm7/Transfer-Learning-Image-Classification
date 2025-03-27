import os
import shutil
import random

def split_dataset(source, destination, split_ratio=0.8):
    train_dir = os.path.join(destination, "train")
    val_dir = os.path.join(destination, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(source):
        class_path = os.path.join(source, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            split_point = int(split_ratio * len(images))
            train_imgs = images[:split_point]
            val_imgs = images[split_point:]

            for mode, img_list in zip([train_dir, val_dir], [train_imgs, val_imgs]):
                os.makedirs(os.path.join(mode, class_name), exist_ok=True)
                for img in img_list:
                    shutil.copy2(os.path.join(class_path, img), os.path.join(mode, class_name, img))
            print(f"{class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

if __name__ == "__main__":
    split_dataset("../data_model", "../separation_donnee", split_ratio=0.8)