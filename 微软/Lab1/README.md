配置Container进行云上训练或推理

通过pytorch提供的模型demo在docker中训练并部署推理服务(torchserve)，同时学习下trition部署推理服务

## 本地测试模型

代码来自pytoch的git项目: https://github.com/pytorch/examples/tree/main/mnist

## 安装docker

[mac安装docker](https://docs.docker.com/desktop/setup/sign-in/)

测试是否安装成功:
```bash
$ docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (arm64v8)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/
```

## 运行容器

Alpine Linux 是一种轻量级、安全、面向容器优化的Linux发行版。它以其小巧的体积和高效性著称，广泛用于Docker容器中，
接下来将在操作系统中运行。

拉取镜像:
```bash
$ docker pull alpine
```


`pull`命令会从**Docker registry**下载镜像，并存在下系统中，可以通过```docker images```命令列出所有镜像
```bash
$ docker images

REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
alpine        latest    44a37b14f342   2 days ago      8.18MB
hello-world   latest    ee301c921b8a   19 months ago   9.14kB
```

### Docker run命令启动容器

基于alpine运行一个Docker container，通过执行```docker run```命令，运行容器:
```bash
$ docker run alpine ls -l
total 56
drwxr-xr-x    2 root     root          4096 Dec  5 12:20 bin
drwxr-xr-x    5 root     root           340 Dec  7 12:49 dev
drwxr-xr-x    1 root     root          4096 Dec  7 12:49 etc
drwxr-xr-x    2 root     root          4096 Dec  5 12:20 home
drwxr-xr-x    6 root     root          4096 Dec  5 12:20 lib
......
......
```

上面的命令在执行完后容器也会停止，所以可以尝试进入交互式模式
```bash
$ docker run -it alpine /bin/sh 
.....
$ exit
```

可以通过```docker ps```列出所有正在运行的容器，当然也可以通过```docker ps -a```列出所有容器

### 容器资源

先启动一个容器：```docker run -it alpine /bin/sh```，然后再打开一个shell窗口，观察容器占用宿主机的资源情况
```bash
$ docker ps
CONTAINER ID   IMAGE     COMMAND     CREATED          STATUS          PORTS     NAMES
dbde346d7e5b   alpine    "/bin/sh"   19 seconds ago   Up 18 seconds             infallible_elgamal

# 资源使用情况
$ docker stats dbde346d7e5b

CONTAINER ID   NAME                 CPU %     MEM USAGE / LIMIT   MEM %     NET I/O       BLOCK I/O    PIDS
dbde346d7e5b   infallible_elgamal   0.00%     428KiB / 7.667GiB   0.01%     1.02kB / 0B   0B / 4.1kB   1

# 查看容器挂载的文件系统
$ docker exec dbde346d7e5b df -h

Filesystem                Size      Used Available Use% Mounted on
overlay                  58.4G      1.6G     53.8G   3% /
tmpfs                    64.0M         0     64.0M   0% /dev
shm                      64.0M         0     64.0M   0% /dev/shm
/dev/vda1                58.4G      1.6G     53.8G   3% /etc/resolv.conf
/dev/vda1                58.4G      1.6G     53.8G   3% /etc/hostname
/dev/vda1                58.4G      1.6G     53.8G   3% /etc/hosts
tmpfs                    64.0M         0     64.0M   0% /proc/kcore
tmpfs                    64.0M         0     64.0M   0% /proc/keys
tmpfs                    64.0M         0     64.0M   0% /proc/timer_list
tmpfs                     3.8G         0      3.8G   0% /sys/firmware

```

## Docker部署Pytorch训练程序


在使用Docker部署应用程序时，一般要先写Dockerfile，通过其构建镜像，然后启动容器。

### 构建Dockerfile

Docker会按顺序通过Dockerfile在构建阶段执行一系列Docker命令:
```bash
# 基础镜像
FROM nvidia/cuda:12.6.3-base-ubuntu20.04

# 创建镜像中的文件夹，用于存储新的代码或文件
RUN mkdir -p /src/app

# WORKDIR指令设置Dockerfile中的任何RUN,CMD,ENTRPOINT,COPY和ADD指令的工作目录
WORKDIR /src/app

# 拷贝本地文件到Docker镜像中相应目录
COPY mnist/main.py /src/app

ENV CONDA_AUTO_YES=true

# 需要安装的依赖
RUN apt-get update && apt-get install wget -y
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH

RUN pip install torch torchvision

# 容器启动命令
CMD ["python", "main.py"]
```

### 构建Docker镜像

上面构建了Dockerfile文件，接下来通过下面的命令来构建Docker镜像:
```bash
docker build -f Dockerfile.gpu -t train_dl .
```

`docker build` : 通过Dockerfile进而构建镜像的命令。

` -t` : 't'代表标签。用户通过标签未来可以确定相应的镜像。

`train_dl` : 给镜像打上的标签名字。

` . ` : 希望构建进镜像中的Dockerfile的相对路径。

` -f ` : 指定构建进镜像中的Dockerfile。

` Dockerfile.gpu` : 构建进镜像中的Dockerfile的文件名，如果机器没有GPU，可以用Dockerfile.cpu文件。

最后看下创建好的镜像文件:
```bash
$ docker images
REPOSITORY    TAG                       IMAGE ID       CREATED         SIZE
train_dl      latest                    d0a65a9b1b17   7 seconds ago   1.68GB
alpine        latest                    44a37b14f342   2 days ago      8.18MB
nvidia/cuda   12.6.3-base-ubuntu20.04   58e13fba4d04   2 weeks ago     238MB
hello-world   latest                    ee301c921b8a   19 months ago   9.14kB
```

### 启动镜像

启动刚才构建的镜像:
```bash
docker run --name training train_dl

......
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.329474
Train Epoch: 1 [640/60000 (1%)] Loss: 1.422901
Train Epoch: 1 [1280/60000 (2%)]        Loss: 1.004950
Train Epoch: 1 [1920/60000 (3%)]        Loss: 0.605173
Train Epoch: 1 [2560/60000 (4%)]        Loss: 0.456048
......
```

## Docker部署Pytorch推理程序

### 创建TorchServer镜像

准备TorchServe源码:
```bash
git clone https://github.com/pytorch/serve.git
```

创建基于CPU镜像

```bash
$ docker build --file Dockerfile.infer.cpu -t torchserve:0.1-cpu .
$ docker images
REPOSITORY    TAG                       IMAGE ID       CREATED          SIZE
torchserve    0.1-cpu                   4ca9cef15321   15 seconds ago   2.01GB
```

### 使用TorchServe镜像启动一个容器

启动CPU容器,打开8080/81端口，并暴露给主机

```bash
$ docker run --rm -it -p 8080:8080 -p 8081:8081 torchserve:0.1-cpu
```

在相同的主机，访问TorchServe APIs, 可以通过主机的8080和8081端口访问
```bash
# 访问服务
$ curl http://localhost:8080/ping

{
  "status": "Healthy"
}
```

检查容器的端口映射
```bash
$ docker port 3775ce21d2ef

# 返回结果
8080/tcp -> 0.0.0.0:8080
8081/tcp -> 0.0.0.0:8081
```

停止Torchserve容器
```bash
$ docker container stop 3775ce21d2ef
```

### 部署模型，进行推理

使用TorchServe推理，第一步需要把模型使用model archiver归档为MAR文件

这里使用的镜像是Dockerfile.tmp构建的镜像，models文件夹为模型文件和Python代码  

模型文件通过在本地下载:
```bash
$ wget https://download.pytorch.org/models/densenet161-8d451a50.pth
```

python代码为[model.py](models/model.py)

启动镜像后，进入容器:
```bash
# 查看容器id
$ docker ps

CONTAINER ID   IMAGE            COMMAND                  CREATED          STATUS         PORTS                                                       NAMES
8877fe474c08   torchserve:tmp   "/usr/local/bin/dock…"   11 seconds ago   Up 10 seconds   7070-7071/tcp, 8082/tcp, 0.0.0.0:8080-8081->8080-8081/tcp   cool_hodgkin

# 进入容器
$ docker exec -it 8877fe474c08 /bin/bash

$ ll

total 113068
drwxr-xr-x 1 model-server model-server      4096 Dec 12 14:24 ./
drwxr-xr-x 1 root         root              4096 Sep 30 21:43 ../
-rw-r--r-- 1 model-server model-server       220 Feb 25  2020 .bash_logout
-rw-r--r-- 1 model-server model-server      3771 Feb 25  2020 .bashrc
-rw-r--r-- 1 model-server model-server       807 Feb 25  2020 .profile
-rw-rw-r-- 1 root         root               309 Sep 30 21:38 config.properties
-rw-r--r-- 1 root         root         115730790 Dec 12 14:18 densenet161-8d451a50.pth
drwxr-xr-x 3 model-server model-server      4096 Dec 12 14:24 logs/
drwxr-xr-x 2 model-server root              4096 Sep 30 21:43 model-store/
-rw-r--r-- 1 root         root              1093 Dec 12 13:51 model.py
drwxr-xr-x 1 model-server root              4096 Dec 12 14:24 tmp/
```

使用model archiver进行模型归档
```bash
torch-model-archiver --model-name densenet161 --version 1.0 --model-file model.py --serialized-file densenet161-8d451a50.pth --export-path /home/model-server/model-store --handler image_classifier
```

启动Torchserve进行推理模型
```bash
# 先停止Torchserve
$ torchserve --stop
TorchServe has stopped.

# 然后再次启动torchserve
$ torchserve --start --ncs --model-store model-store --models densenet161.mar
```

成功启动打印了一些日志
```bash
Torchserve version: 0.12.0
TS Home: /home/venv/lib/python3.9/site-packages
Current directory: /home/model-server
Temp directory: /home/model-server/tmp
Metrics config path: /home/venv/lib/python3.9/site-packages/ts/configs/metrics.yaml
Number of GPUs: 0
Number of CPUs: 5
Max heap size: 1964 M
Python executable: /home/venv/bin/python
Config file: config.properties
Inference address: http://0.0.0.0:8080
Management address: http://0.0.0.0:8081
Metrics address: http://0.0.0.0:8082
Model Store: /home/model-server/model-store
Initial Models: densenet161.mar
Log dir: /home/model-server/logs
Metrics dir: /home/model-server/logs
Netty threads: 32
Netty client threads: 0
......
```

使用模型进行推理:
- 开启新的终端窗口
- 使用`curl`命令下载一个实例并且通过表示将其重命名
- 使用`curl`发送kitten图像，post到Torchserve的predict终端入口

```bash
$ curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
$ curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg
```

结果出现了400，权限问题:
```bash
{
  "code": 400,
  "type": "InvalidKeyException",
  "message": "Token Authorization failed. Token either incorrect, expired, or not provided correctly"
}
```


