## 從github clone下，到cd到train_pcl_api路徑  
```
cd AUO_PCL_FASTAPI/train_pcl_api
```

## 建立image，在terminal輸入
```
docker build -t pcl_train_image .
```

## 建立container，在terminal輸入
```
docker run -d --gpus all --shm-size 4G --name pcl_train_container -p 8000:8000 pcl_train_image
```

## 在網址欄中輸入就可以進到api網頁
host_url:8000/docs  
e.g. http://hc8.isl.lab.nycu.edu.tw:8000/docs

## 查看logs，在terminal輸入
```
docker logs pcl_train_container
```
