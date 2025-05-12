# Base Images 
## 从天池基础镜像构建(from的base img 根据自己的需要更换，建议使用天池open list镜像链接：https://tianchi.aliyun.com/forum/postDetail?postId=67720) 
FROM registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10


## 把当前文件夹里的文件构建到镜像的根目录下,并设置为默认工作目录 
ADD . /app

WORKDIR /app

##安装python依赖包 
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt

## 镜像启动后统一执行 sh run.sh 
CMD ["sh", "/app/run.sh"]
