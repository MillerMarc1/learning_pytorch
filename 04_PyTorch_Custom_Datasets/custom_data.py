import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchinfo import summary

import os
import pathlib
import random
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from timeit import default_timer as timer 


def main():
    # Setup device-agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Get all image paths (* means "any combination")
    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # Write transform for image
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])

    def plot_transformed_images(image_paths, transform, n=3, seed=42):
        """Plots a series of random images from image_paths.

        Will open n image paths from image_paths, transform them
        with transform and plot them side by side.

        Args:
            image_paths (list): List of target image paths. 
            transform (PyTorch Transforms): Transforms to apply to images.
            n (int, optional): Number of images to plot. Defaults to 3.
            seed (int, optional): Random seed for the random generator. Defaults to 42.
        """
        random.seed(seed)
        random_image_paths = random.sample(image_paths, k=n)
        for image_path in random_image_paths:
            with Image.open(image_path) as f:
                fig, ax = plt.subplots(1, 2)
                ax[0].imshow(f) 
                ax[0].set_title(f"Original \nSize: {f.size}")
                ax[0].axis("off")

                # Transform and plot image
                # Note: permute() will change shape of image to suit matplotlib 
                # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
                transformed_image = transform(f).permute(1, 2, 0) 
                ax[1].imshow(transformed_image) 
                ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
                ax[1].axis("off")

                fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

        plt.show()

    plot_transformed_images(image_path_list, 
                            transform=data_transform, 
                            n=3)







    # Option 1: Loading Image Data Using ImageFolder

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=data_transform, # transforms to perform on data (images)
                                    target_transform=None,) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=data_transform
                                    )

    print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

    # Get class names as a list
    class_names = train_data.classes
    print(class_names)

    # Can also get class names as a dict
    class_dict = train_data.class_to_idx
    print(class_dict)

    # Check the lengths
    print(len(train_data), len(test_data))

    img, label = train_data[0][0], train_data[0][1]
    print(f"Image tensor:\n{img}")
    print(f"Image shape: {img.shape}")
    print(f"Image datatype: {img.dtype}")
    print(f"Image label: {label}")
    print(f"Label datatype: {type(label)}")

    # Rearrange the order of dimensions
    img_permute = img.permute(1, 2, 0)

    # Print out different shapes (before and after permute)
    print(f"Original shape: {img.shape} -> [color_channels, height, width]")
    print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

    # Plot the image
    plt.figure(figsize=(10, 7))
    plt.imshow(img_permute)
    plt.axis("off")
    plt.title(class_names[label], fontsize=14)
    plt.show()

    # Turn train and test Datasets into DataLoaders
    BATCH_SIZE=1
    train_dataloader = DataLoader(dataset=train_data, 
                                batch_size=BATCH_SIZE, # how many samples per batch?
                                num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=BATCH_SIZE, 
                                num_workers=1, 
                                shuffle=False) # don't usually need to shuffle testing data

    print(train_dataloader, test_dataloader)



    img, label = next(iter(train_dataloader))

    # Batch size will now be 1, try changing the batch_size parameter above and see what happens
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")

if __name__ == '__main__':
    main()