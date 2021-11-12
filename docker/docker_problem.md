1.Add docker to sudo  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-00-33.png)
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-04-03.png)

权限拒绝，说明上面操作有问题  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-06-45.png)  
不加`sudo`docker push不上去(Dockerhub)，仍然是权限问题  

![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-07-10.png)  
`sudo -i`  
进入根用户  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-07-23.png)
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-07-51.png)  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-07-51.png)  
仍被拒绝  
![](https://github.com/weiweia92/pictures/blob/master/Screenshot%20from%202020-07-02%2020-15-57.png)
