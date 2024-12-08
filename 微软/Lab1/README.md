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


