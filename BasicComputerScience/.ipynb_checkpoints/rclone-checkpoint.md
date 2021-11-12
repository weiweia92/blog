### rclone  
1.Download the bash script and let it does the install :  

`curl https://rclone.org/install.sh | sudo bash`  

![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-44-55.png)
2.Install from source  

`git clone https://github.com/rclone/rclone.git`  
`cd rclone`  
`go build`  
`./rclone version`  

3.把安装rclone的操作步骤在本地写成一个shell脚本  
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-45-17.png)
将文件变成可执行文件  

`chmod +x rclone.sh`  

然后在本地运行scp将shell脚本远程传到服务器上，因为这个安装脚本很小，所以传输很快，没关系，但每重新建一次服务器都要这样操作一次  

`scp -P <PORT> <LOCAL_FILE> root@<REMOTE_IP>:<REMOTE_DEST_DIR>`　

`eg: scp -P 18024 /home/weiweia92/rclone.sh root@ssh4.vast.ai:/root/`   

在服务器上运行rclone.sh  

`./rclone.sh`   

![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-45-30.png)
此例中name命名为nyudrive,一般我命名为Google_Drive  
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-45-41.png)
storage填google drive对应的数字(注意:这个数字每次都变，动态的)  
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-45-51.png)
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-46-01.png)
复制link后面的网址，在网页上打开以获取验证码  
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-46-14.png)
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-46-23.png)
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-46-32.png)
将验证码复制粘贴到之前的Enter verification code>后面  
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-46-40.png)
可以看见已经将远程的google drive挂载到服务器上了    
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-46-49.png)
在服务器上mkdir一个文件夹，这个文件夹用于放即将要从谷歌云下载的东西    
`rclone sync Google_Drive:yolov4-vast/ /root/yolov4-vast`  
![](https://github.com/weiweia92/pictures/blob/master/rclone/Screenshot%20from%202020-07-04%2010-47-05.png)

4.rclone command lines  


`rclone listremotes`:列出所有远程网盘名称   

`rclone lsd Google_Drive`:列出Google_Drive网盘下的所有文件  

`rclone mkdir Google_Drive:rclone-test-folder`:在名为Google_Drive的网盘下建立一个文件夹名字为rclone-test-folder  

`rclone copy <remote file_path> <google drive name>:<google drive dir path>`  

`eg:rclone copy /root/best.weights Google_Drive:weights_folder`:将服务器上的/root/路径下的best.weights文件传到google drive上的weights_folder文件夹里  

详情请看:https://rclone.org/docs/  
