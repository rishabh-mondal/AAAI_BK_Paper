import os
import numpy as np
import h5py
from PIL import Image
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            # img = Image.open(img_path)
            # images.append(np.array(img))
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            images.append(image)
    return images
folder_path = 'PATH TO THE FOLDER WITH IMAGES'
images = load_images_from_folder(folder_path)
# Save to HDF5 file
with h5py.File('user_images.h5', 'w') as f:
    f.create_dataset('image', data=images)
    # f.create_dataset('label', data=all_labels)