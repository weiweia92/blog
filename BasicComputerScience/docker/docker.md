## Docker  

### Docker Images  

1.Read Only Template Used To Create Containers  

2.Build By Docker User  

3.Stored In Docker Hub Or Your Local Registry  

### Docker Container  

Docker Images --> run to generate Docker Containers  

1.Isolated Application Platform  

2.Contains Everything Needed To Run The Application  

3.Built From One Or More Images  

- `docker container logs [CONTAINER ID]`:come out the container logs  

- `docker container kill`:

- `docker container rm [CONTAINER ID]`:remove the docker container  

- `docker container run`   

- `docker container start`  

## Docker Commands  

How to add docker to sudo:  

`sudo groupadd docker`  

`sudo gpasswd -a weiweia92 docker`  

`newgrp docker`  

- `sudo apt-get install docker.io`:install docker  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-01%2009-42-41.png)  

Install successfully!  

- `docker --version`:returns the version of Docker which is installed  


We need to start the docker service after install.  

- `sudo service docker start`    

- `docker info`:查看docker信息  

此时的docker是一个‘裸’docker，没有container也没有image，image简单理解就是container的只读版，用来方便存储与交流  

- `docker pull ubuntu:18.04`:pulls a new Docker image from the Docker Hub  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-01%2009-43-02.png)

`-it`:表示交互  
仔细看，你会发现终端交互的用户名变掉了，说明我们进入到了容器的内部,执行`exit`退出此终端，回到系统本身的终端  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-01%2009-43-23.png)

- `docker ps`:This command lists all the running containers in the host  

- `docker ps -a`:If '-a' flag is specified,shutdown containers are alse displayed  

- `sudo docker commit -m "Added ubuntu18.04" -a "weiweia92" 79c761f627f3 weiweia92/ubuntu:v1`  

`commit`命令用来将容器转化为镜像  
`m`用来来指定提交的说明信息;`-a`可以指定用户信息的;`79c761f627f3`代表容器的`id`;`weiweia92/ubuntu:v1`指定目标镜像的用户名、仓库名和 `tag` 信息  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-01%2009-44-00.png)

- `docker images`:lists down all the images in your local repo  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-01%2009-44-12.png)  

- `docker login`:The command is used to Login to Docker Hub repo from the CLI  

- `docker push`:This command pushed a Docker image on your local repo to the Docker Hub  

- `docker run -it test:0.1`:This command executes a Docker image on your local repo&creates a running Container out of it  

- `docker rm <CONTAINER ID>`:remove the container

- `docker rmi -f <IMAGE>`:remove docker image  

- `docker build -t <REPOSITORY>:<TAG> .`:This command is used to compile the Dockerfile,for building custom Docker images  

- `docker stop <CONTAINER ID>`:This command shuts down the container whose Container ID is specified in arguments.Container is shut down gradfully by waiting for other dependencies to shut, not a force stop  

- `docker kill <CONTAINER ID>`:This command kills the container by stopping its execution immediately.Its similar to force kill.  

- `docker exec`:This command is used to access an already running container and perform operations inside the container.  

`eg:docker exec -it fe6e370a1c9c bash`  

- `docker commit`:This command creats a new image of an edited container on the local repo  

`eg:docker commit fe6e370a1c9c vardhanns/MyModifiedImage`  

- `docker export --output=<'filename.tar'> <CONTAINER ID/IMAGE NAME>`:This command is used to export a Docker image into a tar file in your local system.  

`eg:docker export --output='latest.tar' mycontainer`   

- `docker import <tar file path>`:This command is used to import the contents of a tar file(usually a Docker image)into your local repo.  

`eg:docker import /home/weiweia92/Downloads/demo.tgz`  


- `docker search 关键字`  

`eg：docker search redis`,检索镜像(一般从docker hub检索)  

- `docker images | grep none | awk '{print $3} ' | xargs docker rmi`:删除所有名称为none的镜像  

### 如何将建立好的镜像上传到dockerhub上  

首先打开Dockerhub登录自己的账号,建立一个新的仓库，仓库的名字要和待上传的镜像名字一致,实例中叫test  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-01%2009-44-23.png)

在终端执行`docker push weiweia92/test:0.1`  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-01%2009-44-32.png)

将yolov4模型的test镜像上传到dockerhub上  


`docker run -it test:0.1`: go to docker image  

Done!  
  

最好建立一个文件夹专门放Dockerfile的  

`mkdir docker-sample && cd docker-sample`  

`touch Dockerfile`
  
## Dockerfile Syntax  

Dockerfile syntax consists of two kind of main line blocks:comments and commands+arguments  

**FROM**   

From directive is probably the most crucial amongst all others for Dockerfiles.It defines the base image to use to start the build process.  

`FROM [base image name]`  

`FROM ubuntu:18.04`  

**RUN**   

RUN command is the central executing directive for Dockerfiles.It takes a command as its argument and runs it to form the image.Unlike CMD,it actually is used to build the image.  

`RUN [command]`  

`RUN apt-get install -y riak`  

It has a slight difference from CMD.RUN is used to run a command,which could be a shell command or basically runs my image into a container.But CMD,it can execute a shell command like `CMD "echo""Welcome to Edureka"`,but however it can not use CMD to build my docker image.  

**ENTRYPOINT**  

**ADD**  

ADD command gets two arguments:a source and a destination.It basically copies the files from the source on the host into the container's own filesystem at the set destination.  

`ADD [source directory or URL] [destination directory]`  

`ADD /my_app_folder /my_app_folder`  

**ENV**  

The ENV command is used to set the environment variables(one or more).These variables consist of "key value"pairs which can be accessed within the container by scripts and applications alike.  

`ENV SERVER_WORKS 4`  

`ENV applocation /usr/src `  

`COPY flask-helloworld $applocation/flask-helloworld`  

`ENV flaskapp $applocation/flask-helloworld`  

**WORKDIR**  

WORKDIR directive is used to set where the command defined with CMD is to be executed. WORKDIR允许你在Docker建立image时更改目录，the new directory remains the current directory for the rest of the build instructions.

`WORKDIR /path`

**CMD and ENTRYPOINT**  

CMD  

CMD instruction runs commands like RUN,but the commands run when the Docker container launches.Only one CMD instruction can be used.  

`CMD ["python", "app.py"]`  

ENTRYPOINT：  
    * 设置容器启动时运行的命令  
    * 让容器以应用程序或者服务的形式运行（后台进程）  
    * 不会被忽略，一定会执行  
    * 常见使用方式：写一个脚本作为ENTRYPOINT去执行  

**EXPOSE**  

EXPOSE command is used to associate a specified port to enable networking between the running process inside the c
ontainer and the outside world.  

`EXPOSE 8080`  

EXPOSE告诉Docker服务端容器暴露的端口号，供互联系统使用。在启动Docker时，可以通过-P,主机会自动分配一个端口号转发到指定的端口，如：

`docker run -d -p 127.0.0.1:23333:22 centos6-ssh`:容器ssh服务的22端口将被映射到宿主机的23333端口  

**MAINTAINER**  

THis non-executing command declares the author,hence setting the author field of the images.It should come nonetheless(尽管如此) after FROM.  

`MAINTAINER [name]`  

`MAINTAINER authors_name`  

**USER**  

The USER directive is used to set the UID(or username)which is to run the container based on the image being built.  

`USER [UID]`  

`USER 751`  

### 根据Dockerfile创建docker image  

建立完一个sample_image/Dockerfile后，创建docker image  

`sudo docker build -t sample_image .`  

### 运行Docker container   

`sudo docker run -ip 5000:5000 sample_image:latest`  

-i:交互  

-p:docker host port:Docker container port  

