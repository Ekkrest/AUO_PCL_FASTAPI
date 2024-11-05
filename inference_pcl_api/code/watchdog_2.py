from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import *
import argparse
import json
import requests
import subprocess
import os
import signal
import shutil
import csv
import yaml

class FileEventHandler(FileSystemEventHandler):
    def __init__(self, callback_url, j_id, tj_id):
        self.callback_url = callback_url
        self.j_id = j_id
        self.tj_id = tj_id

    def on_any_event(self, event):
        pass
    
    def on_moved(self, event):
        pass
    
    def on_created(self, event):
        pass

    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        # check if the training is completed
        with open('./status.json', 'r') as f:
            status = json.load(f)

        # callback function
        if self.callback_url is not None:
            print(self.callback_url)
            response = requests.post(self.callback_url, data = json.dumps(status), headers = {'Content-Type': 'application/json'})
            print(response.status_code)
            print(response.json())

        # check if the training is completed
        if status['completed'] == True:
            
            print("Training Completed")

            # update the job list
            with open('./jobs/job.json', mode = 'r') as file:
                job_list = json.load(file)
            j_idx = next((i for i in range(len(job_list)) if job_list[i]['job_id'] == self.j_id), None)
            
            # with open("./testjobs/testjob.json", mode='r') as file:
            #     testjob_list = json.load(file)
            # tj_idx = next((i for i in range(len(testjob_list)) if testjob_list[i]['testjob_id'] == int(self.tj_id)), None)
            
            job_list[j_idx]['status'] = "Finished"


            if job_list[j_idx]['type'] == "train":
                complete_train_function(job_list[j_idx], self.j_id)
            else:
                complete_test_function(int(self.tj_id))

            with open('./jobs/job.json', mode = 'w') as file:
                json.dump(job_list, file, ensure_ascii=False, indent=4)
            
            # kill the watchdog process
            signal.raise_signal( signal.SIGINT )

    
def complete_train_function(job_list, job_id):

    # job_path = os.path.join("./jobs", str(job_id))
    # # move the models
    # model_path = os.path.join("./records", job_list["name"], "best.pt")
    # shutil.copy2(model_path, job_path)
    
    # # save test path to yaml (for inference.sh)
    # with open("./configs/config.yaml", 'r') as rfile:
    #     cfg = yaml.safe_load(rfile)

    # cfg['eval_model'] = os.path.join("./jobs", str(job_id))

    # with open("./configs/config.yaml", 'w') as file:
    #     yaml.dump(cfg, file, sort_keys=False)

    # modelcfg_path = os.path.join("./records", job_list["name"], "config.yaml")
    # shutil.copy2(modelcfg_path, job_path)
    
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

    # copy the training result to the job folder
    # modellog_path = os.path.join("./records", job_list["name"], "logger.log")
    # shutil.copy2(modellog_path, job_path)
    

        

def complete_test_function(testjob_id):
    
    testjob_path = os.path.join("./testjobs", str(testjob_id))
    # move the models
    resultCP_path = os.path.join("./result", "classification_report.csv")
    shutil.copy2(resultCP_path, testjob_path)

    resultCM_path = os.path.join("./result", "CM.jpg")
    shutil.copy2(resultCM_path, testjob_path)
    
    # remove test set
    data_list = os.listdir("./testdatas")
    for f in data_list:
        file_path = os.path.join("./testdatas", f)
        if os.path.isdir(file_path):
        # 如果是目錄，使用 shutil.rmtree 刪除整個目錄
            shutil.rmtree(file_path)
        else:
        # 如果是檔案，使用 os.remove 刪除單一檔案
            os.remove(file_path)


if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser(description="Watch dog")
    # parser.add_argument('--j_id', default="", type=int, help="")
    parser.add_argument('--tj_id', default=None, type=int, help="")
    parser.add_argument('--url', default=None, type=str, help="")
    args = parser.parse_args()

    print("Watch Dog Process Start !!!!!!")

    # with open("./jobs/job.json", mode='r') as file:
    #     job_list = json.load(file)
    # j_idx = next((i for i in range(len(job_list)) if job_list[i]['job_id'] == int(args.j_id)), None)
    # task_type = job_list[j_idx]['type']

    # print(os.getcwd())

    with open('./status.json', 'w') as f:
        pass

    pid = subprocess.Popen(["sh", "./run.sh"], shell=False)

    # observer = Observer()
    # event_handler = FileEventHandler(args.url, int(args.j_id), args.tj_id)
    # observer.schedule(event_handler, path='./status.json', recursive=False)
    # observer.start()
    # try:
    #     while True:
    #         time.sleep(5)
    # except KeyboardInterrupt:
    #     observer.stop()
    #     print("Watch Dog Process Stop !!!")
    # observer.join()


    