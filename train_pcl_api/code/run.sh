#!/bin/bash

# 取得 j_id 作為第一個參數
j_id=$1

# 使用 j_id 作為參數傳遞給 Python 腳本
python main_pcl.py --j_id $j_id
