import logging
import torch
from PIL import Image
from torchvision import transforms
import math
import cv2
from tqdm import tqdm
import argparse
import numpy as np

N_BATCHES = 5
N_WORKERS = 18


def open_image(path: str, resize: float = 1.0) -> Image.Image:
    img = cv2.imread(path)
    return cv2.resize(img, dsize=None, fx=resize, fy=resize)

class _BackgroundSubtractorDataset(torch.utils.data.Dataset):
    def calc_background_subtract(self, image):
        mask = cv2.cvtColor((self.fgbg.apply(image) > 1).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # カーネルで表示領域を増やす
        kernel = np.ones((5, 5), np.float32) / 25
        mask = cv2.filter2D(mask, -1, kernel)

        response = torch.stack(
            [
                torch.tensor(mask * image, dtype=torch.uint8),
                torch.tensor((1 - mask) * image, dtype=torch.uint8),
            ])
        return response
    
    def __init__(self, images, F):
        self.images = images
        self.F = F
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        
        for image in images:
            self.fgbg.apply(image)
    
    def __getitem__(self, idx):
        return self.calc_background_subtract(self.images[idx])
    
    def __len__(self):
        return len(self.images)


class DataSet(torch.utils.data.Dataset):        
    def __init__(self, 
                 paths_image, 
                 labels, 
                 F=16, 
                 resize=1.0, 
                 is_video=True,
                 visualize=False,
                  n_batches=N_BATCHES,
                  n_workers=N_WORKERS):
        self.F = F
        self.is_video = is_video
        self.labels = labels
        
        # Function to transform images into features
        if visualize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.images = [
            open_image(path, resize=resize)
            for path in paths_image
        ]
        loader = torch.utils.data.DataLoader(
            _BackgroundSubtractorDataset(self.images, F),
            shuffle=False,
            batch_size=n_batches,
            num_workers=n_workers)
        self.masked_images = torch.stack([
            torch.stack([
                self.transform(Image.fromarray(row[0].numpy())),
                self.transform(Image.fromarray(row[1].numpy()))
            ])
        for batches in loader
        for row in batches])
        
        self.labels = labels    
        self.func_labels = (lambda X: 1 if sum(X) > 0 else 0) if is_video else (lambda X: X[0])

    def __len__(self):
        if self.is_video:
            return self.masked_images.__len__() // self.F + 1
        else:
            return self.masked_images.__len__()            

    def _get_start_and_end(self, idx):        
        if self.is_video:
            return idx * self.F, idx  * self.F + self.F
        else:
            return idx, idx + self.F

    def __getitem__(self, idx):
        start, end = self._get_start_and_end(idx)
        
        if start > self.__len__() - self.F and self.is_video:
            # Count from the end and match if the number is not divisible.
            start = self.__len__() - self.F
            end = self.__len__()
        sub_images = self._select(self.masked_images[start:end])
        sub_labels = self.labels[start:end]
        return sub_images, self.func_labels(sub_labels)
       
    def _select(self, images):
        if self.is_video:
            return images.transpose_(0, 1)
        else:
            return images[:1].transpose_(0, 1)