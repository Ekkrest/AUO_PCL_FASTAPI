import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from modules import VectorQuantizedVAE
import cv2
import numpy as np
import torch.nn.functional as F
from argparse import Namespace
import yaml
import shutil
from customDataset import InferenceDataset
from torch.utils.data import DataLoader
import Res2netFFM
from torch.nn import DataParallel
import Identity
from sklearn.cluster import KMeans
import csv
from multiprocessing import Process, Array, Lock, Manager
import json

lock = Lock()
# 轉換影像的變換過程
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
def dict_to_namespace(d):
    return Namespace(**d)

def _extract_features(model, dataloader, device):
    all_features = []
    filenames_list = []
    with torch.no_grad():
        for batch in dataloader:
            images, filenames = batch
            images = images.to(device)
            features = model(images)
            all_features.append(features)
            filenames_list.extend(filenames)  # 收集所有檔案名稱
    return torch.cat(all_features), filenames_list

def _load_model_from_path(model_path, device):
    model = Res2netFFM.Res2Net(block=Res2netFFM.Bottle2neck, layers=[3, 4, 6, 3], num_classes=128)
    try:
        model = model.to(device)  # Move model to device first
        if device == 'cuda':
            device_ids = list(range(torch.cuda.device_count()))
            parallel_model = DataParallel(model, device_ids=device_ids)
        else:
            parallel_model = model  # 在 CPU 上，不需要使用 DataParallel

        loaded_state = torch.load(model_path, map_location=device)
        state_dict = loaded_state['state_dict']

        # 修改 key 的名稱以符合當前模型
        for k in list(state_dict.keys()):
            if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
                new_key = k[len("module.encoder_q."):]
                state_dict[new_key] = state_dict[k]
            del state_dict[k]   
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key.replace("encoder.", "")
                new_state_dict[new_key] = value       

        # 加載模型參數
        parallel_model.load_state_dict(new_state_dict, strict=False)
        parallel_model.fc = Identity.Identity()
        parallel_model = parallel_model.to(device)
        parallel_model.eval()
        return parallel_model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
def _write_to_csv(csv_file_path, data, mode='w'):
    with open(csv_file_path, mode=mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def main(args):
        # Prepare the dataset using the extracted images
    test_dataset = InferenceDataset(
        folder_path1 = args.defect_img_folder,
        folder_path2 = args.defect_mask_folder,
        transform=transform
    )
    dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    model = _load_model_from_path(args.moco_model_path, device=args.device)
    # 提取特徵
    all_features, filenames = _extract_features(model, dataloader, args.device)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=args.n_clusters)
    # 如果使用 CPU，確保所有張量都在 CPU 上
    if args.device == 'cpu':
        labels = kmeans.fit_predict(all_features.cpu().numpy())
    else:
        labels = kmeans.fit_predict(all_features.cuda().cpu().numpy())

    # Adjust labels to start from 1 instead of 0
    labels = [label + 1 for label in labels]

    # Prepare the result data for writing to CSV
    output_csv_path = f"./result/{args.result_name}.csv"
    result_data = [["filename", "cluster_label"]]
    result_data.extend([[filenames[i], labels[i]] for i in range(len(filenames))])
    _write_to_csv(output_csv_path, result_data)
    inference_data = os.path.join('./dataset', args.exp_name)
    print(inference_data)
    shutil.rmtree(inference_data)  # Remove the extracted directory
    with lock:
        init_status = dict()
        init_status['completed'] = True
        with open('./status.json', 'w') as f:
            json.dump(init_status, f)

if __name__ == '__main__':
    # 讀取 YAML 配置文件
    with open('./vqvae_config/config.yaml', 'r') as file:
        args = yaml.safe_load(file)
        args = dict_to_namespace(args)
        print(args)
    main(args)