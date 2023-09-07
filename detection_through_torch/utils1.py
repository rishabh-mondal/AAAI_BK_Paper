import numpy as np
import matplotlib.pyplot as plt
import random
import h5py
import torch
import torch.nn.functional as F
import torch.nn as nn

try:
    import torchvision.models as models
except ImportError:
    get_ipython().system('%pip install torch torchvision')
    import torchvision.models as models


def imgs_input_fn(images):
    img_size = (224, 224, 3)
    images = torch.tensor(images)
    images = F.interpolate(images, size=img_size[:2])
    return images

def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        images = np.array(h5_file['image'])
        labels = np.array(h5_file['label'])    
    print("Images shape: ", images.shape) #, 'Images dtype: ', images.dtype)
    print("Labels shape: ", labels.shape) #, 'Labels dtype: ', labels.dtype)
    return images, labels

def show_random_images_with_labels(hdf5_file, num_images=20):
    with h5py.File(hdf5_file, "r") as f:
        images = f["image"]
        labels = f["label"]
        total_images = images.shape[0]
        random.seed(42)
        random_indices = random.sample(range(total_images), num_images)
        num_rows = (num_images + 4) // 5  
        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 3*num_rows))
        for i, index in enumerate(random_indices):
            row = i // 5
            col = i % 5
            image = images[index]
            label = labels[index]
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_title(f"Label: {int(label)}")
            ax.axis("off")
        for i in range(num_images % 5, 5):
            axes[num_rows - 1, i].axis("off")
        plt.tight_layout()
        plt.show()

def densenet_model(learning_rate=0.00002,fine_tune=True,seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = models.densenet121(pretrained=True)
    if not fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.classifier.in_features
    model.classifier =nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5,inplace=True),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    model.eval()
    return model, optimizer, criterion 
