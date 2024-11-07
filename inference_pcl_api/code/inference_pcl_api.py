# FastAPI
from typing import Union
from enum import Enum
from fastapi import FastAPI, File, UploadFile, Response, status, BackgroundTasks
from fastapi.responses import FileResponse
import cv2
import numpy as np
import json
import zipfile
import os
import aiofiles
import shutil
import sys
import csv
import torch
import yaml
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import timm
from torch.nn import DataParallel
import uvicorn
import logging
from multiprocessing import Process, Array, Lock, Manager
import io
import time
import Identity
import subprocess
import Res2netFFM
from customDataset import InferenceDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

app = FastAPI()
lock = Lock()
download_dir = "./download"
weight_dir = "./weights"
vqvae_weight_dir = './vqvae_weights'
weight_info = "./weights/info.json"
vqvae_weight_info = './vqvae_weights/info.json'
result_dir = "./result"
data_dir = './data'
vqvae_data_dir = './vqvae_data'
image_type = ('.png', '.jpg', '.jpeg', '.bmp')
weight_type = ('h5', 'ckpt', 'pth', 'pt', 'tar')

vqvae_manager = Manager()
moco_manager = Manager()
share_moco_models = moco_manager.list([])
share_vqvae_models = vqvae_manager.list([])
process_model = [{'model_id' : None, 'model' : None}]
vqvae_process_model = [{'model_id' : None, 'model' : None}]


class Device(str, Enum):
    cpu = "cpu"
    gpu = "gpu"

# 轉換影像的變換過程
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def _critical_section_updata_weight_list(weight_list_id, w):
    with lock:
        # print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        logger.info(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        weight_list = _get_weight_list()
        weight_list[weight_list_id] = w
        _updata_weight_list(weight_list)
        logger.info(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
        # print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")

def _critical_section_updata_vqvae_weight_list(weight_list_id, w):
    with lock:
        # print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        logger.info(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        weight_list = _get_vqvae_weight_list()
        weight_list[weight_list_id] = w 
        _updata_vqvae_weight_list(weight_list)
        logger.info(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
        # print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")

def _critical_section_weight_id():
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        w = {"weight_id": None, "name": None, "info": None, "file_name": None, "file_path" : None}
        # get the current weight list
        weight_list = _get_weight_list()
        # assign weight_id
        if len(weight_list):
            w["weight_id"] = weight_list[-1]["weight_id"] + 1
        else:
            w["weight_id"] = 0
        # add into weight list
        weight_list.append(w)
        # update the weight_list
        _updata_weight_list(weight_list)
        weight_list_id,_ = _get_weight_index(w["weight_id"])
        print("weight_id : " + str(w["weight_id"]) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return weight_list_id, w

def _critical_section_vqvae_weight_id():
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        w = {"weight_id": None, "name": None, "info": None, "file_name": None, "file_path" : None}
        # get the current weight list
        weight_list = _get_vqvae_weight_list()
        # assign weight_id
        if len(weight_list):
            w["weight_id"] = weight_list[-1]["weight_id"] + 1
        else:
            w["weight_id"] = 0
        # add into weight list
        weight_list.append(w)
        # update the weight_list
        _updata_vqvae_weight_list(weight_list)
        weight_list_id,_ = _get_vqvae_weight_index(w["weight_id"])
        print("weight_id : " + str(w["weight_id"]) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return weight_list_id, w

def _critical_section_share_models(model_info):
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        if len(share_moco_models) == 0:
            model_info['model_id'] = 0
            share_moco_models.append(model_info)
        else:
            model_info['model_id'] = int(len(share_moco_models))
            share_moco_models.append(model_info)
        print("model ID " + str(model_info['model_id']) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return model_info['model_id']

def _critical_section_vqvae_share_models(model_info):
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        if len(share_vqvae_models) == 0:
            model_info['model_id'] = 0
            share_vqvae_models.append(model_info)
        else:
            model_info['model_id'] = int(len(share_vqvae_models))
            share_vqvae_models.append(model_info)
        print("model ID " + str(model_info['model_id']) + " is created.")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")
    return model_info['model_id']

def _get_weight_list():
    if not os.path.isfile(weight_info):
        return []
    else:
        with open(weight_info, mode='r') as file:
            weight_list = json.load(file)
        return weight_list
    
def _get_vqvae_weight_list():
    if not os.path.isfile(vqvae_weight_info):
        return []
    else:
        with open(vqvae_weight_info, mode='r') as file:
            weight_list = json.load(file)
        return weight_list
    
def _updata_weight_list(weight_list):
    with open(weight_info, mode='w') as file:
        json.dump(weight_list, file, ensure_ascii=False, indent=4)
    logger.info("weight list update!")
    # print("weight list update!")
 
def _updata_vqvae_weight_list(weight_list):
    with open(vqvae_weight_info, mode='w') as file:
        json.dump(weight_list, file, ensure_ascii=False, indent=4)
    logger.info("weight list update!")
    # print("weight list update!")


def _load_model_from_path(model_path, device):
    model = Res2netFFM.Res2Net(block=Res2netFFM.Bottle2neck, layers=[3, 4, 6, 3], num_classes=128)
    try:
        model = model.to(device)  # Move model to device first
        if device.type == 'cuda':
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


def _inference(imgs, model, best_cls):
    cls_name = str(best_cls[:-6]) #刪除'-top-1' 
    with torch.no_grad():
        if cls_name == 'combiner':
            logits = model(imgs)[cls_name]
        else:
            logits = model(imgs)[cls_name].mean(1)
    
    pred = torch.max(logits, dim=-1)[1]
    score = torch.max(logits, dim=-1)[0]
    return pred, score

def _check_weight_list(weight_list, weight_id):
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return False
    else:
        return True

def _get_weight_index(weight_id):
    weight_list = _get_weight_list()
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return w_idx, None
    else:
        return w_idx, weight_list[w_idx]
    
def _get_vqvae_weight_index(weight_id):
    weight_list = _get_vqvae_weight_list()
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return w_idx, None
    else:
        return w_idx, weight_list[w_idx]

def _write_to_csv(csv_file_path, data, mode='w'):
    with open(csv_file_path, mode=mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def _delayed_remove(path: str, delay: int = 10):
    time.sleep(delay)
    os.remove(path)

@app.get("/weight/")
async def get_weight_list(response: Response):
    """Return weight list

    Args:
        response (Response): response

    Returns:
        list: [dict: weight_list, int: error_code]
    """
    weight_list = _get_weight_list()

    error_code = 0
    logger.info("get_weight_list!")
    return {"weight_list": weight_list, "error_code": error_code}

@app.get("/vqvae_weight/")
async def get_vqvae_weight_list(response: Response):
    """Return vqvae weight list

    Args:
        response (Response): response

    Returns:
        list: [dict: weight_list, int: error_code]
    """
    weight_list = _get_vqvae_weight_list()

    error_code = 0
    logger.info("get_weight_list!")
    return {"weight_list": weight_list, "error_code": error_code}

@app.post("/weight/{name}")
async def post_weight(response: Response, weight: UploadFile, name: str, info: Union[str, None] = None):
    """Received the zip file of the Weights, create weight_info, 
    assign weight_id, add weight_info into record(json file),
    store the weights into weight folder by the weight_id

    Args:
        response (Response): response
        name (str): the name of the weights
        info (sre): the annotation of the weights
        weight (UploadFile): the zip file of the weights

    Returns:
        list: [int: weight_id, int: error_code]
    """
    weight_list_id, w = _critical_section_weight_id()
    w["name"] = name
    w["info"] = info
    # Store the zip
    zip_path = os.path.join(weight_dir, "{}.zip".format(str(w["weight_id"])))

    async with aiofiles.open(zip_path, mode="wb") as out_file:
        content = await weight.read()
        await out_file.write(content)

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Upload file is not a zip file."}

    # Find Zip Fild name
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    # Extract the zip
    w_dir_path = os.path.join(weight_dir, str(w["weight_id"]))
    if not os.path.exists(w_dir_path):
        os.mkdir(w_dir_path)

    with zipfile.ZipFile(zip_path, mode='r') as zip_file:
        zip_file.extractall(w_dir_path)

    # 獲取解壓後的內容（資料夾或檔案）
    extracted_items = os.listdir(w_dir_path)

    # 假設解壓後的內容只有一個資料夾 a，要將該資料夾的內容移動到 w_dir_path
    if len(extracted_items) == 1:
        extracted_folder = os.path.join(w_dir_path, extracted_items[0])
        if os.path.isdir(extracted_folder):
            # 如果是資料夾，將該資料夾的內容移動到 w_dir_path
            for item in os.listdir(extracted_folder):
                source_path = os.path.join(extracted_folder, item)
                destination_path = os.path.join(w_dir_path, item)
                shutil.move(source_path, destination_path)
            # 刪除多餘的資料夾 a
            os.rmdir(extracted_folder)
            
    # Delete zip
    os.remove(zip_path)
    
    w["file_path"] = w_dir_path
    for root, dirs, files in os.walk(w_dir_path):
        for f in files:
            print(f)
            if f.endswith(weight_type):
                w["file_name"] = f
    # Update the weight_list
    _critical_section_updata_weight_list(weight_list_id, w)
    
    error_code = 0
    logger.info("post_weight!")
    return {"weight_id": w["weight_id"], "error_code": error_code}

@app.post("/vqvae_weight/{name}")
async def post__vqvae_weight(response: Response, weight: UploadFile, name: str, info: Union[str, None] = None):
    """Received the zip file of the Weights, create weight_info, 
    assign weight_id, add weight_info into record(json file),
    store the weights into weight folder by the weight_id

    Args:
        response (Response): response
        name (str): the name of the weights
        info (sre): the annotation of the weights
        weight (UploadFile): the zip file of the weights

    Returns:
        list: [int: weight_id, int: error_code]
    """
    weight_list_id, w = _critical_section_vqvae_weight_id()
    w["name"] = name
    w["info"] = info
    # Store the zip
    zip_path = os.path.join(vqvae_weight_dir, "{}.zip".format(str(w["weight_id"])))

    async with aiofiles.open(zip_path, mode="wb") as out_file:
        content = await weight.read()
        await out_file.write(content)

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Upload file is not a zip file."}

    # Find Zip Fild name
    zip_name = os.path.splitext(os.path.basename(zip_path))[0]
    # Extract the zip
    w_dir_path = os.path.join(vqvae_weight_dir, str(w["weight_id"]))
    if not os.path.exists(w_dir_path):
        os.mkdir(w_dir_path)

    with zipfile.ZipFile(zip_path, mode='r') as zip_file:
        zip_file.extractall(w_dir_path)

    # 獲取解壓後的內容（資料夾或檔案）
    extracted_items = os.listdir(w_dir_path)

    # 假設解壓後的內容只有一個資料夾 a，要將該資料夾的內容移動到 w_dir_path
    if len(extracted_items) == 1:
        extracted_folder = os.path.join(w_dir_path, extracted_items[0])
        if os.path.isdir(extracted_folder):
            # 如果是資料夾，將該資料夾的內容移動到 w_dir_path
            for item in os.listdir(extracted_folder):
                source_path = os.path.join(extracted_folder, item)
                destination_path = os.path.join(w_dir_path, item)
                shutil.move(source_path, destination_path)
            # 刪除多餘的資料夾 a
            os.rmdir(extracted_folder)
            
    # Delete zip
    os.remove(zip_path)
    
    w["file_path"] = w_dir_path
    for root, dirs, files in os.walk(w_dir_path):
        for f in files:
            print(f)
            if f.endswith(weight_type):
                w["file_name"] = f
    # Update the weight_list
    _critical_section_updata_vqvae_weight_list(weight_list_id, w)
    
    error_code = 0
    logger.info("post_weight!")
    return {"weight_id": w["weight_id"], "error_code": error_code}


@app.delete("/weight/{weight_id}")
async def delete_weight(response: Response, weight_id: int):
    """Delete the weights of weight_id

    Args:
        response (Response): response
        weight_id (int): the weights of weight_id to be deleted

    Returns:
        int: error_code
    """
    error_code = 0
    weight_list = _get_weight_list()
    w_idx, _ = _get_weight_index(weight_id)
    if w_idx is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Model is not exist."}
    else:
        del weight_list[w_idx]
        _updata_weight_list(weight_list)
        shutil.rmtree(os.path.join(weight_dir, str(weight_id)))
        logger.info("delete_weight!")
        

    return {"error_code": error_code}

@app.delete("/vqvae_weight/{weight_id}")
async def delete_weight(response: Response, weight_id: int):
    """Delete the weights of weight_id

    Args:
        response (Response): response
        weight_id (int): the weights of weight_id to be deleted

    Returns:
        int: error_code
    """
    error_code = 0
    weight_list = _get_vqvae_weight_list()
    w_idx, _ = _get_vqvae_weight_index(weight_id)
    if w_idx is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Model is not exist."}
    else:
        del weight_list[w_idx]
        _updata_vqvae_weight_list(weight_list)
        shutil.rmtree(os.path.join(vqvae_weight_dir, str(weight_id)))
        logger.info("delete_weight!")
        

    return {"error_code": error_code}

@app.post("/weight/load/{weight_id}")
async def post_load_weight(response: Response, weight_id: int):
    """Load the weight of weight_id

    Args:
        response (Response): response
        weight_id (int): the weight of weight_id to be loaded
    Returns:
        int: error_code
    """

    weight_list = _get_weight_list()

    if _check_weight_list(weight_list, weight_id):
        id, weight_list = _get_weight_index(weight_id)
        model_info={
            "model_id": None,
            "name": str(weight_list['name']),
            "model_path": str(os.path.join(weight_list['file_path'], weight_list['file_name'])),
        }
        model_id = _critical_section_share_models(model_info)
    else:
        return {"error_code": 1, "error_msg": "Model is not exist."}

    return {"error_code": 0, "model_id": model_id}

@app.post("/vqvae_weight/load/{weight_id}")
async def post_load_vqvae_weight(response: Response, weight_id: int):
    """Load the weight of weight_id

    Args:
        response (Response): response
        weight_id (int): the weight of weight_id to be loaded
    Returns:
        int: error_code
    """

    weight_list = _get_vqvae_weight_list()

    if _check_weight_list(weight_list, weight_id):
        id, weight_list = _get_vqvae_weight_index(weight_id)
        model_info={
            "model_id": None,
            "name": str(weight_list['name']),
            "model_path": str(os.path.join(weight_list['file_path'], weight_list['file_name'])),
        }
        model_id = _critical_section_vqvae_share_models(model_info)
    else:
        return {"error_code": 1, "error_msg": "Model is not exist."}

    return {"error_code": 0, "model_id": model_id}

@app.get("/weight/load/")
async def get_load_weight(response: Response):
    """Get the model list
    Args:
        response (Response): response
    Returns:
        loaded models: models_list, int: error_code
    """
    pid = os.getpid()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logger.info(f"[{timestamp}] Process {pid} is using get_load_weight function.")
    
    models_list = []
    for i in share_moco_models:
        temp = {"model_id" : i['model_id'], "name" :str(i['name']), "model_path": str(i['model_path']) }
        models_list.append(temp) 
    return {"loaded models": models_list, "error_code": 0, "pid": pid}

@app.get("/vqvae_weight/load/")
async def get_load_vqvae_weight(response: Response):
    """Get the model list
    Args:
        response (Response): response
    Returns:
        loaded models: models_list, int: error_code
    """
    models_list = []
    for i in share_vqvae_models:
        temp = {"model_id" : i['model_id'], "name" :str(i['name']), "model_path": str(i['model_path']) }
        models_list.append(temp) 
    return {"loaded models": models_list, "error_code": 0}

def find(weight_id):
    model_exist = next((item for item in share_moco_models if item["model_id"] == weight_id), None)
    return model_exist

def vqvae_find(weight_id):
    model_exist = next((item for item in share_vqvae_models if item["model_id"] == weight_id), None)
    return model_exist

def delete(weight_id):
    model_index = next((id for id, item in enumerate(share_moco_models) if item["model_id"] == weight_id), None)
    del share_moco_models[model_index]

def vqvae_delete(weight_id):
    model_index = next((id for id, item in enumerate(share_vqvae_models) if item["model_id"] == weight_id), None)
    del share_moco_models[model_index]

@app.delete("/weight/load/{weight_id}")
async def delete_load_weight(response: Response, model_id: int):
    """Unload the model

    Args:
        response (Response): response

    Returns:
        int: error_code
    """

    model_exist = find(model_id)
    if model_exist is not None:
        delete(model_id)
        global process_model
        process_model = share_moco_models[:]
        print(len(process_model))
        torch.cuda.empty_cache()
        logger.info("delete_load_weight!")
        return {"error_code": 0}
    else:
        return {"error_code": 1, "error_code": "Model is not loaded in the memory."}
    
@app.delete("/vqvae_weight/load/{weight_id}")
async def delete_load_vqvae_weight(response: Response, model_id: int):
    """Unload the model

    Args:
        response (Response): response

    Returns:
        int: error_code
    """

    model_exist = vqvae_find(model_id)
    if model_exist is not None:
        vqvae_delete(model_id)
        global vqvae_process_model
        vqvae_process_model = share_vqvae_models[:]
        print(len(vqvae_process_model))
        torch.cuda.empty_cache()
        logger.info("delete_load_weight!")
        return {"error_code": 0}
    else:
        return {"error_code": 1, "error_code": "Model is not loaded in the memory."}

@app.post("/inference/batch")
async def Inference_Batch(response: Response, background_tasks: BackgroundTasks, file: UploadFile, model_id: int, n_clusters: int = 4, device: Device = "cpu", result_name: Union[str, None] = None):
    """Inference a batch of images with loaded model (model_id)

    Args:
        response (Response): HTTP response
        file (UploadFile): A batch of images compressed by zip
        model_id (int): loaded model to used
        device : cpu / gpu
    Returns:
        return: the cluster result of the batch of images.
    """

    inference_start = time.time()
    share_model_index = next((id for id, item in enumerate(share_moco_models) if item["model_id"] == model_id), None)
    if share_model_index is None:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Model is not exist."}
    
    copy_end = time.time()

    if device == 'cpu':
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")

    model_info = share_moco_models[share_model_index]
    print(model_info)
    model = _load_model_from_path(model_info["model_path"], device=device)
    if model is None:
        return {"error_code": 3, "error_msg": "Failed to load the model."}

    # model = model.to(device)
    filename = file.filename
    if not filename.lower().endswith('zip'):
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 2, "error_msg": "Upload file is not a valid file, only zip format."}

    zip_save_path = os.path.join(data_dir, filename)
    extract_dir = os.path.join(data_dir, os.path.splitext(filename)[0])

    # Save the uploaded zip file to the data directory
    async with aiofiles.open(zip_save_path, mode="wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Extract the zip file into the designated directory
    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    base_name = os.path.splitext(filename)[0]

    # Prepare the dataset using the extracted images
    test_dataset = InferenceDataset(
        folder_path1=os.path.join(extract_dir, 'defect_img'),
        folder_path2=os.path.join(extract_dir, 'defect_mask'),
        transform=transform
    )
    dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # 提取特徵
    all_features, filenames = _extract_features(model, dataloader, device)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    # 如果使用 CPU，確保所有張量都在 CPU 上
    if device.type == 'cpu':
        labels = kmeans.fit_predict(all_features.cpu().numpy())
    else:
        labels = kmeans.fit_predict(all_features.cuda().cpu().numpy())

    # Adjust labels to start from 1 instead of 0
    labels = [label + 1 for label in labels]

    # Prepare the result data for writing to CSV
    output_csv_path = f"./result/{result_name}.csv"
    result_data = [["filename", "cluster_label"]]
    result_data.extend([[filenames[i], labels[i]] for i in range(len(filenames))])
    _write_to_csv(output_csv_path, result_data)

    # Prepare the JSON result to be returned
    result = [{"input": filenames[i], "predict label": int(labels[i])} for i in range(len(filenames))]
    result.append({"error_code": 0})
    inference_end = time.time()

    # Clean up: remove the extracted files and the uploaded zip file
    shutil.rmtree(extract_dir)  # Remove the extracted directory
    os.remove(zip_save_path)  # Remove the saved zip file

    background_tasks.add_task(_delayed_remove, output_csv_path, delay=100)

    # print("copy time : ", (copy_end - inference_start))
    # print("inference time : ", (inference_end - inference_start))
    logger.info("copy time : %s", (copy_end - inference_start))
    logger.info("inference time : %s", (inference_end - inference_start))
    logger.info("Inference_Batch Finished!")

    return FileResponse(output_csv_path, media_type='application/octet-stream', filename=f"{result_name}.csv")

@app.post("/inference/all_in_one")
async def post_inference(response: Response, 
    background_tasks: BackgroundTasks,
    name: str, data: UploadFile, 
    vqvae_model_id: int,
    moco_model_id: int, 
    n_clusters: int = 4, 
    device: Device = "cpu", 
    hidden_size: Union[int, None] = None,
    k: Union[int, None] = None,
    batch_size: Union[int, None] = None,
    result_name: Union[str, None] = None
    ):
    """ Post the Image Folder dataset to start a training

    Args:
        response: Http Response
        name: the name of this trainig task
        weight_id: if you need use the pretrained model to train, fill this parameter.
        batch_size: hyper-parameter to be modified
        workers: hyper-parameter to be modified
        epochs: hyper-parameter to be modified

    Returns:
        job_id: the job_id of the trianing process

    """
    with lock:

        share_model_index = next((id for id, item in enumerate(share_vqvae_models) if item["model_id"] == vqvae_model_id), None)
        moco_share_model_index = next((id for id, item in enumerate(share_moco_models) if item["model_id"] == moco_model_id), None)
        model_info = share_vqvae_models[share_model_index]
        moco_model_info = share_moco_models[moco_share_model_index]
        print(model_info)
        print(moco_model_info)    
        print("step 1. Weight management passed")
        # Download the dataset
        if not os.path.exists(vqvae_data_dir):
            os.mkdir(vqvae_data_dir)

        data_zip_path = os.path.join(vqvae_data_dir, str(data.filename))
        async with aiofiles.open(data_zip_path, mode="wb") as out_file:
            print(data_zip_path)
            content = await data.read()
            await out_file.write(content)
        
        # Check if it is a zip file
        if not zipfile.is_zipfile(data_zip_path):
            os.remove(data_zip_path)
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error_code": 3, "error_msg": "Upload file is not a zip file."}
        
        # Extract files
        with zipfile.ZipFile(data_zip_path, mode='r') as zip_file:
            zip_file.extractall(vqvae_data_dir)
        os.remove(data_zip_path)
        data_path, _ = os.path.splitext(data_zip_path)

        print("step 2. Dataset managemet passed")
        if device == 'cpu':
            device = "cpu"
        else:
            if torch.cuda.is_available():
                device = "cuda"
        
        with open('./vqvae_config/config.yaml', 'r') as f:
                config = yaml.load(f, Loader = yaml.FullLoader)

        config['exp_name'] = name if name is not None else config['exp_name']
        config['data_dir'] = data_path
        config['batch_size'] = batch_size if batch_size is not None else config['batch_size']
        config['hidden_size'] = hidden_size if hidden_size is not None else config['hidden_size']
        config['k'] = k if k is not None else config['k']
        config['model_path'] = model_info["model_path"]
        config['moco_model_path'] = moco_model_info['model_path']
        config['n_clusters'] = n_clusters if n_clusters is not None else config['n_clusters']
        config['device'] = device if device is not None else config['device']
        config['defect_img_folder'] = os.path.join('./dataset', name, 'defect_img') 
        config['defect_mask_folder'] = os.path.join('./dataset', name, 'defect_mask') 
        config['result_name'] = result_name if result_name is not None else config['result_name']
        with open('./vqvae_config/config.yaml', 'w') as f:
            yaml.dump(config, f, sort_keys=False)

        print("step 3. Config management passed")

        # Call the watch dog program
        proc = subprocess.Popen(["python", "watchdog_2.py"], shell=False, preexec_fn=os.setsid)
        print("step 4. call watch_dog.py")
        
        # 使用 communicate() 確保進程完成執行
        stdout, stderr = proc.communicate()

        # 檢查進程的返回碼是否為 0，表示成功執行
        if proc.returncode != 0:
            print(f"watchdog_2.py 執行失敗: {stderr}")
            response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            return {"error_code": 5, "error_msg": "Error occurred during execution of watchdog_2.py."}

        # Clean up: remove the extracted files and the uploaded zip file
        #shutil.rmtree(vqvae_data_dir)  # Remove the extracted directory
        # 定義重試機制，等待csv檔案完成
        max_retries = 10  # 重試次數
        wait_time = 10    # 每次重試等待的秒數
        output_csv_path = f"./result/{result_name}.csv"
        time.sleep(wait_time)

        with open('./status.json', 'r') as f:
            idle = json.load(f)
        
        for _ in range(max_retries):
            with open('./status.json', 'r') as f:
                idle = json.load(f)
            if idle['completed'] == True:
                background_tasks.add_task(_delayed_remove, output_csv_path, delay=100)
                return FileResponse(output_csv_path, media_type='application/octet-stream', filename=f"{result_name}.csv")
            else:
                # 如果檔案還沒生成，等待並重試
                time.sleep(wait_time)

        # 若嘗試數次後檔案依然不存在，返回錯誤訊息
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 4, "error_msg": "Zip file not found after waiting."}

if __name__ == "__main__":
    logger.info("Fast API Activate !!!")
