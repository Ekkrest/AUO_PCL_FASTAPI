from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv
import torch
import os

class InferenceDataset(Dataset):
    def __init__(self, folder_path1, folder_path2, transform=None):
        self.folder_path1 = folder_path1
        self.folder_path2 = folder_path2
        self.image_files1 = sorted(os.listdir(folder_path1))
        self.image_files2 = sorted(os.listdir(folder_path2))
        self.transform = transform

    def __len__(self):
        return len(self.image_files1)

    def __getitem__(self, index):
        
        image1_path = os.path.join(self.folder_path1, self.image_files1[index])
        #print(len(image1_path))

        # Assuming the second image has a similar filename pattern,
        # you may need to adjust this based on your actual file naming convention.
        image2_path = os.path.join(self.folder_path2, self.image_files2[index])

        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        pair1 = torch.cat((img1, img2), dim=0)

        return pair1, os.path.basename(image1_path)        