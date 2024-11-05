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

def load_model(args):
    # 加載模型
    model = VectorQuantizedVAE(input_dim=1, dim=args.hidden_size, K=args.k).to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    return model

def preprocess_image(image_path,args):
    # 預處理圖片
    img_size = 256
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    image = Image.open(image_path)
    image_gray = image.convert('L')
    image = transform(image).unsqueeze(0)
    image = image.to(args.device)
    image_gray = transform(image_gray).unsqueeze(0)
    image_gray = image_gray.to(args.device)
    return image, image_gray

def preprocess_image_to_1024(image_path):
    # 預處理圖片
    img_size = 1024
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = np.array(image)

    return image

def image_to_defect_patch(origin, reconstruct, origin_rgb,origin_result_folder,position_result_folder, filename):
    # 計算差異
    diff = cv2.absdiff(origin, reconstruct)

    diff = cv2.medianBlur(diff, 3)
    # 進行二值化處理
    adaptive_thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, -10)

    # 取得圖片尺寸
    height, width = adaptive_thresh.shape

    # 設定保留區域的左上角和右下角座標
    start_x = (width - 236) // 2
    start_y = (height - 236) // 2
    end_x = start_x + 236
    end_y = start_y + 236

    # 將保留區域以外的部分設定為黑色
    result = adaptive_thresh.copy()
    result[:start_y, :] = 0
    result[end_y:, :] = 0
    result[:, :start_x] = 0
    result[:, end_x:] = 0

    # 將瑕疵位置圖放大為1024*1024
    result_large = cv2.resize(result, (1024, 1024))
    max_sum = -1
    max_patch = None
    max_origin_patch = None

    # 將放大後的瑕疵位置圖切成256*256共16張，並儲存包含瑕疵的patch
    for i in range(4):
        for j in range(4):
            patch = result_large[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]
            patch_sum = np.sum(patch)
            # 保留最大總和的patch
            if patch_sum > max_sum:
                max_sum = patch_sum
                max_patch = patch
                max_origin_patch = origin_rgb[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, :]
    if max_patch is not None:
        origin_save_path = os.path.join(origin_result_folder, filename)
        position_save_path = os.path.join(position_result_folder, filename)
        cv2.imwrite(origin_save_path, max_origin_patch)
        cv2.imwrite(position_save_path, max_patch)

        
        
def dict_to_namespace(d):
    return Namespace(**d)

def main(args):
    count = [0]

    # 加載模型
    model = load_model(args)
    model.eval()
    
    defect_img_folder = os.path.join('./dataset',args.exp_name,'defect_img')
    defect_mask_folder = os.path.join('./dataset',args.exp_name,'defect_mask')

    if not os.path.exists(defect_img_folder):
        os.makedirs(defect_img_folder)
    if not os.path.exists(defect_mask_folder):
        os.makedirs(defect_mask_folder)

    # 遍歷資料夾中的圖片並進行測試
    for filename in os.listdir(args.data_dir):
        image_path = os.path.join(args.data_dir, filename)
        image, image_gray = preprocess_image(image_path, args)
        image1024 = preprocess_image_to_1024(image_path)

        with torch.no_grad():
            reconstructed_image, _, _ = model(image_gray)


        origin_numpy_image = (image_gray.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5
        origin_numpy_image = origin_numpy_image.astype(np.uint8)

        rgb_numpy_image = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5
        rgb_numpy_image = rgb_numpy_image.astype(np.uint8)

        reconstructed_numpy_image = (reconstructed_image.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1) * 127.5
        reconstructed_numpy_image = reconstructed_numpy_image.astype(np.uint8)
            
        image1024 = cv2.cvtColor(image1024, cv2.COLOR_BGR2RGB)

        image_to_defect_patch(origin_numpy_image,reconstructed_numpy_image,image1024,defect_img_folder,defect_mask_folder,filename)


    

if __name__ == '__main__':
    # 讀取 YAML 配置文件
    with open('./vqvae_config/config.yaml', 'r') as file:
        args = yaml.safe_load(file)
        args = dict_to_namespace(args)
        print(args)
    main(args)
    shutil.rmtree(args.data_dir) 
