from torchvision import transforms
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Subset
import random
from collections import defaultdict


import os
import logging

log = logging.getLogger(__name__)

# YAML KEYS
TRANSFORMATIONS = "transformations"
TRAINING = "train"
VALIDATION = "val"
PATH = "data_path"
LOADERS = "loaders"
TEST_MODE = "test_mode"
DATA_SIZE = "data_size"

def read_transformations(params: dict) -> list:
    """
    Helper function: Read the transformations defined in data_preparation.yml and convert them to a list


    Returns:
        List of torchvision transformations
    """
    transformations = [transforms.ToTensor()]  # normalized to [0, 1]
    for key, arguments in params[TRANSFORMATIONS].items():
        try:
            method = getattr(transforms, key)
            transformations.append(method(**arguments))
        except AttributeError:
            log.error("Transformation: %s does not exist" % key)
            continue
    return transformations


def compose_transformations(params: dict) -> tuple[transforms.Compose]:
    """
    Read and compose the transformations from data_preparation.yml

    Returns:
        A torchvision transforms.Compose object that can be used directly on datasets
    """
    transformations = transforms.Compose(read_transformations(params))
    return transformations


def create_dataset(
    transformations: transforms.Compose, params: dict
) -> ImageFolder:
    """
    Create dataset and apply transformations

    Args:
    - transformations: transforms.Compose.
        Holds all the desired transformations to be applied on the data

    Returns:
        A torchvision datasets.ImageFolder that holds our transformed dataset
    """
    data_path = params[PATH]
    dataset = ImageFolder(root=data_path, transform=transformations)

    if params.get(TEST_MODE, False):
        dataset = Subset(dataset, random.sample(range(0, len(dataset)), params[DATA_SIZE]))
    log.info("%s: \n\t: %d" %(DATA_SIZE, len(dataset)))

    return dataset


def create_dataset_base(
    transformations: transforms.Compose, params: dict, numpy=False, loader=None, ext=None
) -> DatasetFolder:
    """
    Create dataset and apply transformations

    Args:
    - transformations: transforms.Compose.
        Holds all the desired transformations to be applied on the data

    Returns:
        A torchvision datasets.DatasetFolder that holds our transformed dataset
    """
    data_path = params[PATH]
    if numpy:
        dataset = DatasetFolder(root=data_path, transform=transformations, loader=loader, extensions=ext)
    else:
         dataset = ImageFolder(root=data_path, transform=transformations)
    
    if params.get(TEST_MODE, False):
        dataset = Subset(dataset, random.sample(range(0, len(dataset)), params[DATA_SIZE]))
    log.info("%s: \n\t: %d" %(DATA_SIZE, len(dataset)))

    return dataset


def create_data_loader(dataset, params: dict):
    """
    Create dataloader for our dataset
    """
    args = params[LOADERS]
    loader = DataLoader(dataset, **args)
    return loader

def EDA(dataset, params: dict):
    # Get the class names and number of samples per class
    class_names = dataset.classes
    class_counts = defaultdict(int)

    # Iterate through the dataset and count the samples per class
    for _, class_label in dataset:
        class_counts[class_label] += 1

    # Print the class distribution and number of samples per class
    print("Class Distribution:")
    for class_idx, class_name in enumerate(class_names):
        print(f"Class {class_idx}: {class_name} - {class_counts[class_idx]} samples")