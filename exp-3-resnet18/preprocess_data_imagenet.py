import numpy as np
import os
import random
import shutil
import torch
from argparse import ArgumentParser

def prepare_image_net_data(
    data_path: str = os.path.join('data','image_net'),
    seed: int = 1234,
    ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    TRAIN_RATIO = 0.8

    data_dir = data_path

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


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help = "path to the folder images/", default = os.path.join('data','image_net'))
    args_parser = parser.parse_args()
    prepare_image_net_data(data_path=args_parser.data_path)