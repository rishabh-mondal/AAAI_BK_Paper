import h5py
import random
import numpy as np
from PIL import Image  
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50


# Set a seed for NumPy
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

def imgs_input_fn(images):
    img_size = (224, 224, 3)
    images = tf.convert_to_tensor(value = images)
    images = tf.image.resize(images, size=img_size[:2])
    return images

def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        images = np.array(h5_file['image'])
        labels = np.array(h5_file['label'])    
    print("Images shape: ", images.shape, 'Images dtype: ', images.dtype)
    print("Labels shape: ", labels.shape, 'Labels dtype: ', labels.dtype)
    return images, labels


def resnet_model(learning_rate=0.00002):
    # load model
    model = models.Sequential()
    conv_base = ResNet50(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = False # not fine tuning 
    model.add(conv_base)
    
    model.add(layers.AveragePooling2D(pool_size=(7,7)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    	      optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    return model

def show_random_images_with_labels(hdf5_file, num_images=20):
    with h5py.File(hdf5_file, "r") as f:
        images = f["image"]
        labels = f["label"]
        total_images = images.shape[0]
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
            ax.set_title(f"Label: {label}")
            ax.axis("off")
        for i in range(num_images % 5, 5):
            axes[num_rows - 1, i].axis("off")
        plt.tight_layout()
        plt.show()
