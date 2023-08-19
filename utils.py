import h5py
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
np.random.seed(42)
# Set a seed for TensorFlow
tf.random.set_seed(42)
random.seed(42) 
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16, DenseNet121, ResNet50,EfficientNetB0
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from PIL import Image  
# Set a seed for NumPy

def imgs_input_fn(images):
    img_size = (224, 224, 3)
    images = tf.convert_to_tensor(value = images)
    images = tf.image.resize(images, size=img_size[:2])
    return images

def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        images = np.array(h5_file['image'])
        labels = np.array(h5_file['label'])    
    print("Images shape: ", images.shape) #, 'Images dtype: ', images.dtype)
    print("Labels shape: ", labels.shape) #, 'Labels dtype: ', labels.dtype)
    return images, labels

def VGG_16_transfer_model_author(learning_rate=0.00002, fine_tune = True,seed = 42):
    np.random.seed(seed)
    # Set a seed for TensorFlow
    tf.random.set_seed(seed)
    random.seed(seed)
    # load model 
    model = models.Sequential()
    conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = fine_tune  # Fine-tuning the last few layers3
    model.add(conv_base) 
    model.add(layers.AveragePooling2D(pool_size=(7, 7)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2, seed=seed))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(learning_rate=learning_rate),  # Lower learning rate for fine-tuning
                  metrics=['acc'])
    return model

def resnet_model(learning_rate=0.00002, fine_tune = True, seed=42):
    np.random.seed(seed)
    # Set a seed for TensorFlow
    tf.random.set_seed(seed)
    random.seed(seed)
    # load model
    model = models.Sequential()
    conv_base = ResNet50(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = fine_tune # fine tuning 
    model.add(conv_base)
    
    model.add(layers.AveragePooling2D(pool_size=(7,7)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.2, seed=seed))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
    	      optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    return model

def densenet_model(learning_rate=0.00002 , fine_tune = True, seed=42):
    np.random.seed(seed)
    # Set a seed for TensorFlow
    tf.random.set_seed(seed)
    random.seed(seed)
    # load model
    model = models.Sequential()
    conv_base = DenseNet121(include_top=False,weights='imagenet', input_shape=(224, 224, 3))
    conv_base.trainable = fine_tune # fine tuning the last few layers3
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.AveragePooling2D(pool_size=(7,7)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.2, seed=seed))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
    	      optimizer=Adam(learning_rate=learning_rate), metrics=['acc'])
    return model

def efficientnet_b0_model(learning_rate=0.00002, fine_tune = True, seed = 42):
    np.random.seed(seed)
    # Set a seed for TensorFlow
    tf.random.set_seed(seed)
    random.seed(seed)
    
    model = models.Sequential()
    conv_base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    conv_base.trainable = fine_tune # fine tuning
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    # model.add(layers.Dropout(rate=0.2, seed=42))
    model.add(layers.Dense(256, activation='relu'))  # Adding a Dense layer with 256 units and ReLU activation
    model.add(layers.Dropout(rate= 0.2, seed=seed))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=learning_rate),  # Lower learning rate for fine-tuning
                metrics=['acc'])
    return model

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

# function to visualize the images along with true and predicted labels
def visualize_predictions(images, true_labels, predicted_labels, num_samples=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        true_label = "Brick Kiln" if true_labels[i] == 1 else "Not Brick Kiln"
        predicted_label = "Brick Kiln" if predicted_labels[i] > 0.5 else "Not Brick Kiln"
        plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
        plt.axis("off")
    plt.show()
