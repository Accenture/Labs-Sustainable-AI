import copy
import numpy as np
import os
import random
import shutil

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data


def prepare_data(
    data_path: str,
    archive_name: str,
    seed: int = 1234,
    valid_data_amount: float = 0.9,
    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    data_to_extract = os.path.join(data_path, archive_name)
    datasets.utils.extract_archive(data_to_extract, data_path)

    TRAIN_RATIO = 0.8

    data_name = archive_name.split('.')[0]
    data_dir = os.path.join(data_path, data_name)

    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    classes = os.listdir(images_dir)

    for c in classes:
        class_dir = os.path.join(images_dir, c)
        images = os.listdir(class_dir)
 
        n_train = int(len(images) * TRAIN_RATIO)
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        os.makedirs(os.path.join(train_dir, c), exist_ok = True)
        os.makedirs(os.path.join(test_dir, c), exist_ok = True)
        
        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
            
        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image) 
            shutil.copyfile(image_src, image_dst)

    train_data = datasets.ImageFolder(root = train_dir, 
                                    transform = transforms.ToTensor())

    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in train_data:
        means += torch.mean(img, dim = (1,2))
        stds += torch.std(img, dim = (1,2))

    means /= len(train_data)
    stds /= len(train_data)
        
    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')

    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds= [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.CenterCrop(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means, 
                                                    std = pretrained_stds)
                        ])

    train_data = datasets.ImageFolder(root = train_dir, 
                                    transform = train_transforms)

    test_data = datasets.ImageFolder(root = test_dir, 
                                    transform = test_transforms)

    VALID_RATIO = valid_data_amount

    n_train_examples = int(len(train_data) * VALID_RATIO)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data, 
                                            [n_train_examples, n_valid_examples])

    valid_data = copy.deepcopy(valid_data)
    valid_data.dataset.transform = test_transforms

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(valid_data)}')
    print(f'Number of testing examples: {len(test_data)}')

    return train_data, valid_data, test_data



def prepare_image_net_data(
    data_path: str = 'data',
    folder_name: str = 'image_net',
    seed: int = 1234,
    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    TRAIN_RATIO = 0.8

    data_dir = os.path.join(data_path, folder_name)

    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir) 
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    os.makedirs(train_dir)
    os.makedirs(test_dir)

    classes = os.listdir(images_dir)

    print(classes)

    for c in classes:
        class_dir = os.path.join(images_dir, c)
        images = os.listdir(class_dir)

        # maybe shuffle images
 
        n_train = int(len(images) * TRAIN_RATIO)
        train_images = images[:n_train]
        test_images = images[n_train:]
        
        os.makedirs(os.path.join(train_dir, c), exist_ok = True)
        os.makedirs(os.path.join(test_dir, c), exist_ok = True)
        
        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
            
        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image) 
            shutil.copyfile(image_src, image_dst)
