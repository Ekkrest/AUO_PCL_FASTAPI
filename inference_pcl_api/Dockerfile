# 使用帶有 Conda 的基礎映像
FROM continuumio/miniconda3

# 複製代碼和依賴文件
COPY code /code
COPY environment.yml /environment.yml

# 安裝系統依賴
RUN apt update -y && apt upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# 使用 conda 安裝 faiss 和其餘依賴
RUN conda env create -f /environment.yml

# 確保 Conda 環境中的 gunicorn 在 PATH 中
RUN echo "source activate ffmpcl" > ~/.bashrc
ENV PATH=/opt/conda/envs/ffmpcl/bin:$PATH


# 設置工作目錄和啟動命令
WORKDIR /code
# CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "inference_pcl_api:app"]
CMD ["uvicorn", "inference_pcl_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]