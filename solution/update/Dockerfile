# 使用官方 PyTorch 镜像作为基础镜像（含 CUDA 11.8 和 PyTorch 2.0.0）
FROM registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10


COPY requirements.txt .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install torch==2.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
WORKDIR /app
COPY . /app

RUN chmod +x /app/run.sh

CMD ["sh", "/app/run.sh"]
