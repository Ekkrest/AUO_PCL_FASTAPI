from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import csv
import torch
import os

class CustomDataset(Dataset):
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

        pair1 = torch.cat((img1[0], img2[0]), dim=0)
        pair2 = torch.cat((img1[1], img2[1]), dim=0)    

        pair_list = [pair1, pair2]  

        return pair_list, index

class SingleDataset(Dataset):
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

        return pair1, index

class MiniDataset(Dataset):
    def __init__(self, root_dir, transform1, transform2):
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2
        self.classes = sorted(os.listdir(root_dir))  # 獲取所有資料夾的名稱，作為類別

        self.data = []
        self.labels = []
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                self.data.append(os.path.join(class_dir, file_name))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        img1 = self.transform1(img)
        img2 = self.transform2(img)

        pair = torch.cat([img1, img2], dim=0)    

        return pair, label
    
class MVTecDataset(Dataset):
    def __init__(self, folder_path1, folder_path2, transform1, transform2):
        self.folder_path1 = folder_path1
        self.folder_path2 = folder_path2
        self.folder1_images = sorted(os.listdir(self.folder_path1))
        self.folder2_images = sorted(os.listdir(self.folder_path2))
        self.transform1 = transform1
        self.transform2 = transform2

        assert len(self.folder1_images) == len(self.folder2_images)

    def __len__(self):
        return len(self.folder1_images)

    def __getitem__(self, idx):
        img_name1 = self.folder1_images[idx]
        img_name2 = self.folder2_images[idx]
        
        img1 = Image.open(os.path.join(self.folder_path1, img_name1)).convert('RGB')
        img2 = Image.open(os.path.join(self.folder_path2, img_name2)).convert('RGB')
        
        if self.transform1:
            img1 = self.transform1(img1)
        if self.transform2:
            img2 = self.transform2(img2)
        
        # Concatenate two images along the channel dimension
        concatenated_img = torch.cat((img1, img2), dim=0)
        
        # Label is prefix of img_name1
        label = img_name1.split("_")[0]  # Assuming the prefix is separated by "_"
        class_mapping = {
            'bottle': 1,
            'cable': 2,
            'capsule': 3,
            'carpet': 4,
            'grid': 5,
            'hazelnut': 6,
            'leather': 7,
            'metal': 8,
            'pill': 9,
            'screw': 10,
            'tile': 11,
            'toothbrush': 12,
            'transistor': 13,
            'wood': 14,
            'zipper': 15,
        }

        label_number = class_mapping[label]
        
        return concatenated_img, label_number

class ViSADataset(Dataset):
    def __init__(self, folder_path1, folder_path2, transform1, transform2):
        self.folder_path1 = folder_path1
        self.folder_path2 = folder_path2
        self.folder1_images = sorted(os.listdir(self.folder_path1))
        self.folder2_images = sorted(os.listdir(self.folder_path2))
        self.transform1 = transform1
        self.transform2 = transform2

        assert len(self.folder1_images) == len(self.folder2_images)

    def __len__(self):
        return len(self.folder1_images)

    def __getitem__(self, idx):
        img_name1 = self.folder1_images[idx]
        img_name2 = self.folder2_images[idx]
        
        img1 = Image.open(os.path.join(self.folder_path1, img_name1)).convert('RGB')
        img2 = Image.open(os.path.join(self.folder_path2, img_name2)).convert('RGB')
        
        if self.transform1:
            img1 = self.transform1(img1)
        if self.transform2:
            img2 = self.transform2(img2)
        
        # Concatenate two images along the channel dimension
        concatenated_img = torch.cat((img1, img2), dim=0)
        
        # Label is prefix of img_name1
        label = img_name1.split("_")[0]  # Assuming the prefix is separated by "_"
        class_mapping = {
            'candle': 1,
            'capsules': 2,
            'cashew': 3,
            'chewinggum': 4,
            'macaroni1': 5,
            'macaroni2': 6,
            'fryum': 7,
            'pcb1': 8,
            'pcb2': 9,
            'pcb3': 10,
            'pcb4': 11,
            'pipe': 12,
        }

        label_number = class_mapping[label]
        
        return concatenated_img, label_number
    
class DefectDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = []
        self.root_dir = root_dir
        self.transform = transform
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row if there is one
            for row in reader:
                image_path, label, mask_path = row
                full_image_path = os.path.join(self.root_dir, image_path)
                full_mask_path = os.path.join(self.root_dir, mask_path)
                self.data.append((full_image_path, label, full_mask_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label, mask_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('RGB')  # Assuming mask is a grayscale image

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        pair = torch.cat((image, mask), dim=0)
        """label_dict = {'fabric_border': 0, 'pill_type': 1, 'bent': 2, 'missing': 3, 'thread_side': 4, 'cut_lead': 5, 'manipulated_front': 6, 'liquid': 7, 'contamination': 8, 'print': 9, 'split_teeth': 10, 'cut_inner_insulation': 11, 'missing_cable': 12, 'color': 13, 'fabric_interior': 14, 'faulty_imprint': 15, 'bent_wire': 16, 'poke': 17, 'glue_strip': 18, 'broken_teeth': 19, 'melt': 20, 'poke_insulation': 21, 'broken': 22, 'broken_small': 23, 'missing_wire': 24, 'metal_contamination': 25, 'fold': 26, 'combined': 27, 'crack': 28, 'oil': 29, 'squeezed_teeth': 30, 'squeeze': 31, 'thread_top': 32, 'damaged_case': 33, 'defective': 34, 'scratch_neck': 35, 'misplaced': 36, 'cable_swap': 37, 'rough': 38, 'cut_outer_insulation': 39, 'cut': 40, 'broken_large': 41, 'bent_lead': 42, 'glue': 43, 'hole': 44, 'gray_stroke': 45, 'flip': 46, 'scratch_head': 47, 'thread': 48, 'scratch': 49}"""
        label_dict = {'broken_large': 0, 'broken_small': 1, 'contamination': 2, 'bent_wire': 3, 'cable_swap': 4, 'combined': 5, 'cut_inner_insulation': 6, 'cut_outer_insulation': 7, 'missing_cable': 8, 'missing_wire': 9, 'poke_insulation': 10, 'crack': 11, 'faulty_imprint': 12, 'poke': 13, 'scratch': 14, 'squeeze': 15, 'color': 16, 'cut': 17, 'hole': 18, 'metal_contamination': 19, 'thread': 20, 'bent': 21, 'broken': 22, 'glue': 23, 'print': 24, 'fold': 25, 'flip': 26, 'pill_type': 27, 'manipulated_front': 28, 'scratch_head': 29, 'scratch_neck': 30, 'thread_side': 31, 'thread_top': 32, 'glue_strip': 33, 'gray_stroke': 34, 'oil': 35, 'rough': 36, 'defective': 37, 'bent_lead': 38, 'cut_lead': 39, 'damaged_case': 40, 'misplaced': 41, 'liquid': 42, 'broken_teeth': 43, 'fabric_border': 44, 'fabric_interior': 45, 'split_teeth': 46, 'squeezed_teeth': 47}
        
        label = label_dict[label]

        return pair, label

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