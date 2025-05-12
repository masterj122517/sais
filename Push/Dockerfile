# FROM registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10
FROM sais-public-registry.cn-shanghai.cr.aliyuncs.com/sais-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
WORKDIR /app

# 单独复制 requirements.txt，确保 pip install 只在它变动时重新执行
COPY requirements.txt .

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 再复制剩余代码
ADD . /app

CMD ["sh", "/app/run.sh"]
