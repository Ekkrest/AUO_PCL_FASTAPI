from typing import Union
from enum import Enum
from fastapi import FastAPI, File, UploadFile, Response, status, BackgroundTasks, Request
from fastapi.responses import FileResponse, StreamingResponse
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
import torchvision.transforms as transforms
from PIL import Image
import cv2
import timm
from torch.nn import DataParallel
import uvicorn
import logging
from multiprocessing import Process, Array, Lock, Manager
import io
import time
import signal
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess

lock = Lock()
app = FastAPI()
weight_dir = "./weights"
weight_info = "./weights/info.json"
job_dir = "./jobs"
job_info = "./jobs/job.json"
testjob_dir = "./testjobs"
testjob_info = "./testjobs/testjob.json"
image_type = ('.png', '.jpg', '.jpeg', '.bmp')
weight_type = ('h5', 'ckpt', 'pth', 'pt', '.tar', '.pth.tar')
data_dir = "./data"
testdata_dir = "./testdata"

def _get_weight_list():
    if not os.path.isfile(weight_info):
        return []
    else:
        with open(weight_info, mode='r') as file:
            weight_list = json.load(file)
        return weight_list
    
def _get_job_list():
    if not os.path.isfile(job_info):
        return []
    else:
        with open(job_info, mode='r') as file:
            job_list = json.load(file)
        return job_list

def _get_testjob_list():
    if not os.path.isfile(testjob_info):
        return []
    else:
        with open(testjob_info, mode='r') as file:
            testjob_list = json.load(file)
        return testjob_list

def _updata_weight_list(weight_list):
    with open(weight_info, mode='w') as file:
        json.dump(weight_list, file, ensure_ascii=False, indent=4)
    print("weight list update!")

def _update_job_list(job_list):
    with open(job_info, mode='w') as file:
        json.dump(job_list, file, ensure_ascii=False, indent=4)

def _update_testjob_list(testjob_list):
    with open(testjob_info, mode='w') as file:
        json.dump(testjob_list, file, ensure_ascii=False, indent=4)
        
def _get_weight_index(weight_id):
    weight_list = _get_weight_list()
    w_idx = next((i for i in range(len(weight_list))
                  if weight_list[i]['weight_id'] == weight_id), None)
    if w_idx is None:
        return w_idx, None
    else:
        return w_idx, weight_list[w_idx]

def _get_job_index(job_id):
    job_list = _get_job_list()
    j_idx = next((i for i in range(len(job_list)) if job_list[i]['job_id'] == job_id), None)
    print(j_idx)
    if j_idx is None:
        return j_idx, None
    else:
        return j_idx, job_list[j_idx]

def _get_testjob_index(testjob_id):
    testjob_list = _get_testjob_list()
    print(testjob_list)
    tj_idx = next((i for i in range(len(testjob_list)) if testjob_list[i]['testjob_id'] == testjob_id), None)
    if tj_idx is None:
        return tj_idx, None
    else:
        return tj_idx, testjob_list[tj_idx]
    
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

def _critical_section_updata_weight_list(weight_list_id, w):
    with lock:
        print(f"Process {os.getpid()} is entering the critical section. (PID: {os.getpid()})")
        weight_list = _get_weight_list()
        weight_list[weight_list_id] = w
        _updata_weight_list(weight_list)
        print("weight list updated!")
        print(f"Process {os.getpid()} is leaving the critical section. (PID: {os.getpid()})")

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

    return {"weight_list": weight_list, "error_code": error_code}

@app.post("/weight/{name}")
async def post_weight(response: Response, weight: UploadFile, name: str, info: Union[str, None] = None):
    """Received the zip file of the Weights, create weight_info, 
    assign weight_id, add weight_info into record(json file),
    store the weights into weight folder by the weight_id

    Args:
        response (Response): response
        name (str): the name of the weights
        info (str): the annotation of the weights
        weight (UploadFile): the zip file of the weights

    Returns:
        list: [int: weight_id, int: error_code]
    """
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    weight_list_id, w = _critical_section_weight_id()
    w["name"] = name
    w["info"] = info

    # store the zip
    zip_path = os.path.join(weight_dir, "{}.zip".format(str(w["weight_id"])))

    async with aiofiles.open(zip_path, mode="wb") as out_file:
        content = await weight.read()
        await out_file.write(content)

    if not zipfile.is_zipfile(zip_path):
        os.remove(zip_path)
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 1, "error_msg": "Upload file is not a zip file."}

    # extract the zip
    w_dir_path = os.path.join(weight_dir, str(w["weight_id"]))
    if not os.path.exists(w_dir_path):
        os.mkdir(w_dir_path)
    with zipfile.ZipFile(zip_path, mode='r') as zip_file:
        zip_file.extractall(w_dir_path)
    # delete zip
    os.remove(zip_path)
    
    w["file_path"] = w_dir_path
    for root, dirs, files in os.walk(w_dir_path):
        for f in files:
            print(f)
            if f.endswith(weight_type):
                w["file_name"] = f
    # update the weight_list
    _critical_section_updata_weight_list(weight_list_id, w)
    
    error_code = 0
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

    return {"error_code": error_code}    

@app.post("/train/")
async def post_train(response: Response, 
    name: str, data: UploadFile, 
    info: Union[str, None] = None,
    callback_url: Union[str, None] = None,
    resume: Union[int, None] = None, 
    weight_id: Union[int, None] = None,
    batch_size: Union[int, None] = None,
    workers: Union[int, None] = None,
    epochs: Union[int, None] = None,
    warmup_epoch: Union[int, None] = None,
    num_cluster: Union[str, None] = None,
    world_size: Union[int, None] = None,
    rank: Union[int, None] = None,
    ):
    """ Post the Image Folder dataset to start a training

    Args:
        response: Http Response
        name: the name of this training task
        callback_url: the url to post the training status
        weight_id: if you need use the pretrained model to train, fill this parameter.
        batch_size: hyper-parameter to be modified
        workers: hyper-parameter to be modified
        epochs: hyper-parameter to be modified
        num_cluster: hyper-parameter to be modified


    Returns:
        job_id: the job_id of the training process

    """
    try:
        with lock:
            with open('./status.json', 'r') as f:
                idle = json.load(f)
            
            if idle['idle'] == False:
                return {"error_code": 1, "error_msg": "Another job is training or testing."}

            init_status = dict()
            init_status['epoch'] = 0
            init_status['acc'] = 0
            init_status['status'] = "Train"
            init_status['idle'] = False
            init_status['completed'] = False
            with open('./status.json', 'w') as f:
                json.dump(init_status, f)
    
            if not os.path.exists(job_dir):
                os.mkdir(job_dir)
            job = {"job_id": None, "pid": None, "name": name, "info": info, "type": "train", "status": None}

            if weight_id is not None:
                weight_idx, weight_list = _get_weight_index(weight_id)
                if weight_idx is None:
                    response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                    return {"error_code": 2, "error_msg": "weight {weight_id} is not exist."}
                
            print("step 1. Weight management passed.")

            # Download the dataset
            if not os.path.exists(data_dir):
                os.mkdir(data_dir)

            data_zip_path = os.path.join(data_dir, str(data.filename))
            file_name = str(data.filename)
            file_name = file_name.replace('.zip', '')
            
            async with aiofiles.open(data_zip_path, mode="wb") as out_file:
                content = await data.read()
                await out_file.write(content)
            
            # Check if it is a zip file
            if not zipfile.is_zipfile(data_zip_path):
                os.remove(data_zip_path)
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                return {"error_code": 3, "error_msg": "Upload file is not a zip file."}
            
            # Extract files
            with zipfile.ZipFile(data_zip_path, mode='r') as zip_file:
                zip_file.extractall(data_dir)
            
            os.remove(data_zip_path)


            # 檢查是否存在必要的目錄和文件
            required_directories = [
                os.path.join(data_dir, "non_defect_img"),
                os.path.join(data_dir, "defect_img"),
                os.path.join(data_dir, "non_defect_mask"),
                os.path.join(data_dir, "defect_mask"),
                # 如果有其他必要目錄也可以加入此列表
            ]
            for directory in required_directories:
                if not os.path.exists(directory):
                    with open('./status.json', 'r') as f:
                        init_status = json.load(f)

                    init_status['status'] = 'Stoped'
                    init_status['idle'] = True
                    init_status['completed'] = True
                    
                    with open('./status.json', 'w') as f:
                        json.dump(init_status, f)
                    response.status_code = status.HTTP_400_BAD_REQUEST
                    return {"error_code": 6, "error_msg": f"Missing required directory: {directory}"}



            print("step 2. Unsupervised dataset management passed.")

            # Change the configs
            if weight_id is not None:
                cfg_path = os.path.join(str(weight_list['file_path']), 'config.yaml')
                with open(cfg_path, 'r') as f:
                    config = yaml.load(f, Loader = yaml.FullLoader)
            else :    
                with open('./config/config.yaml', 'r') as f:
                    config = yaml.load(f, Loader = yaml.FullLoader)

            config['exp_name'] = name if name is not None else config['exp_name']
            config['batch_size'] = batch_size if batch_size is not None else config['batch_size']  
            config['workers'] = workers if workers is not None else config['workers']
            config['epochs'] = epochs if epochs is not None else config['epochs']
            config['warmup_epoch'] = warmup_epoch if warmup_epoch is not None else config['warmup_epoch']
            config['num_cluster'] = num_cluster if num_cluster is not None else config['num_cluster']
            config['world_size'] = world_size if world_size is not None else config['world_size']
            config['rank'] = rank if rank is not None else config['rank']

            # 取得目前 train_moco_api.py 檔案的目錄路徑
            current_dir = os.path.dirname(os.path.abspath(__file__))
            temp_dir = os.path.join(current_dir, "data")
            config['data_path'] = f"{temp_dir}"
            config['negative_samples_path'] = f"{temp_dir}"

            if resume is None:
                config['resume'] = ''
            elif isinstance(resume, int):
                weight_path = weight_dir + '/' + str(resume)
                if os.path.exists(weight_path):
                    pth_files = [f for f in os.listdir(weight_path) if f.endswith('.pth.tar')]
                    if pth_files:
                        # 如果有 .pth.tar 檔，將檔案名加入路徑
                        config['resume'] = os.path.join(weight_path, pth_files[0])
                        print(f"Resume path set to: {config['resume']}")
                    else:
                        print(f"No .pth.tar file found in {weight_path}")
                        config['resume'] = weight_path  # 仍將資料夾設定為 resume 路徑     
                else:
                    print('the path to resume is wrong.')
                    config['resume'] = ''
            else:
                print('the path to resume is not exist.')
                config['resume'] = ''    
            
            if weight_id is not None:
                pretrained_path = os.path.join(weight_list['file_path'], weight_list['file_name'])
                config['pretrained'] = pretrained_path
            else:
                config['pretrained'] = None
            config['image_root'] = ''
            
            with open('./config/config.yaml', 'w') as f:
                yaml.dump(config, f, sort_keys=False)

            print("step 3. Config management passed")

            job_list = _get_job_list()
            if len(job_list) != 0:
                job["job_id"] = job_list[-1]["job_id"] + 1
            else:
                job["job_id"] = 0
            job["status"] = "running"
            job_list.append(job)
            job_path = os.path.join(job_dir, str(job["job_id"]))
            if not os.path.exists(job_path):
                os.mkdir(job_path)
            _update_job_list(job_list)
            # Call the watch dog program
            if callback_url is None:
                proc = subprocess.Popen(["python", "watchdog_2.py", "--j_id", str(job["job_id"])], shell=False, preexec_fn=os.setsid)
            else:
                proc = subprocess.Popen(["python", "watchdog_2.py", "--j_id", str(job["job_id"]), "--url", callback_url], shell=False, preexec_fn=os.setsid)

            print("step 4. call watch_dog.py")

            # Store the pid into job
            job_list = _get_job_list()
            current_job_id = next((i for i in range(len(job_list)) if job_list[i]['job_id'] == job["job_id"]), None)
            job_list[current_job_id]["pid"] = proc.pid
            _update_job_list(job_list)
            
            print("step 5. job management passed")

        return {"job_id": job["job_id"], "error_code": 0}    
    except Exception as e:
        with open('./status.json', 'r') as f:
            init_status = json.load(f)

        init_status['idle'] = True
        init_status['completed'] = True
        
        with open('./status.json', 'w') as f:
            json.dump(init_status, f)

        print("An error occurred:", str(e))
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"error_code": 5, "error_msg": str(e)}



@app.get("/train/")
async def get_train(response: Response):
    """Get the list of the job_list

    Args:
        response: Http response

    Return:
        the list of the job_list
    """
    # get the job list
    job_list = _get_job_list()

    return {"job_list": job_list, "error_code": 0}
    
@app.get("/train_status")
async def get_train_status(response: Response):
    """Get the training status of the given job_id
    
    Args:
        response: Http response
    Return:
        the training status
    """

    with open("./status.json", mode='r') as f:
        training_status = json.load(f)
    return {"job_info": training_status, "error_code": 0}

@app.delete("/train/{job_id}")
async def delete_train(response: Response, job_id: int):
    """Stop the training of the job_id

    Args:
        response: Http response
        job_id: the job_id to be stopped
    Return:
        the error_code and the error message
    """

    # kill the process
    j_idx, j = _get_job_index(job_id)
    if j_idx is None:
        return {"error_code": 1, "error_msg": "job {job_id} is not exists."}
    elif j["type"] != "train":
        return {"error_code": 2, "error_msg": "job {job_id} is not a training task."}
    else:
        if j["status"] == "Finished":
            return {"error_code": 3, "error_msg": "job {job_id} is finished."}
        # stop the watch_dog process
        os.killpg(os.getpgid(j["pid"]), signal.SIGTERM)
        # delete the dataset
        data_list = os.listdir("./data")
        for f in data_list:
            file_path = os.path.join("./data", f)
            if os.path.isdir(file_path):
            # 如果是目錄，使用 shutil.rmtree 刪除整個目錄
                shutil.rmtree(file_path)
            else:
            # 如果是檔案，使用 os.remove 刪除單一檔案
                os.remove(file_path)

        # update job list
        job_list = _get_job_list()
        del job_list[j_idx]
        _update_job_list(job_list)
        shutil.rmtree(os.path.join(job_dir, str(j["job_id"])))

        status = dict()
        status['status'] = "Delete"
        status['idle'] = True
        status['completed'] = True
        with open('./status.json', 'w') as f:
            json.dump(status, f)

        return {"error_code": 0}    

@app.get("/train/result/{job_id}")
async def get_train_result(response: Response, background_tasks: BackgroundTasks, job_id: int):
    """ Get the training result of the finished training.

    Args:
        response: Http response
        background_tasks: the action after the return the function
        job_id: the job_id to get the result

    Return:
        the zip file of the training result: config(yaml), model weight(.pth.tar)

    """
    # get the training results: config(yaml), model weight(.pth.tar)

    j_idx, j = _get_job_index(job_id)
    if j_idx is None:
        return {"error_code": 1, "error_msg": "job {job_id} is not exists"}
    else:
        j_dir_path = os.path.join(job_dir, str(job_id))
        zip_path = os.path.join(job_dir, "{}.zip".format(str(job_id)))

    if j["status"] != "Finished":
        return {"error_code": 2, "error_msg": "job {job_id} has not finished yet"}

    if os.path.isdir(j_dir_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zfile:
            files = os.listdir(j_dir_path)
            for f in files:
                zfile.write(os.path.join(j_dir_path, f), f)

        background_tasks.add_task(_delayed_remove, zip_path, delay=100)

        return FileResponse(zip_path, media_type='application/zip', filename=f"{job_id}.zip")

@app.delete("/train/result/{job_id}")
async def delete_train_result(response: Response, job_id: int):
    """ Detele the job by the job_id

    Args:
        response: Http resposne
        job_id: the job_id to be deleted

    Return:
        the error code and the error message
    """

    job_list = _get_job_list()
    j_idx, _ = _get_job_index(job_id)

    if j_idx is None:
        return {"error_code": 1, "error_msg": "Job {job_id} is not exists."}
    else:
        if job_list[j_idx]["type"] != "train":
            return {"error_code": 2, "error_msg": "Job {job_id} is not for training."}
        else:
            del job_list[j_idx]
            _update_job_list(job_list)
            shutil.rmtree(os.path.join(job_dir, str(job_id)))
            return {"error_code": 0}