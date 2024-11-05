#!/bin/bash

# 執行inference
python inference.py 
# 等 inference 執行完後，執行 moco_inference
python moco_inference.py