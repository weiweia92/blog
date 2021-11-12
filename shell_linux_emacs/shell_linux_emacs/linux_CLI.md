## Copying files and directories

copy a file named file.txt to file named file2.txt in the current directory.If the destination file exists, it will be overwritten.   

`cp file.txt file2.txt`   

`-i:interactive`  
`-p:preserve the file mode`  

`eg: cp -i file.txt file_backup.txt`  

`uname -a :system info--all`  

`scp`  

## Group

Linux groups are organization units that are used to organize and administer user accounts in Linux. The primary purpose of groups is to define a set of privileges such as reading(r), writing(w), or executing(x) permission for a given resource that can be shared among the users within the group.  

There are two types of groups in Linux operating systems:  
Primary group - When a user creates a file, the file’s group is set to the user’s primary group. Usually, the name of the group is the same as the name of the user. The information about the user’s primary group is stored in the /etc/passwd file.  

Secondary or supplementary group - Useful when you want to grant certain file permissions to a set of users who are members of the group. For example, if you add a specific user to the docker group, the user will inherit the access rights from the group, and be able to run docker commands.  

Only root or users with sudo access can add a user to a group  

### add an existing user to group/multi_group  

`sudo username -a -G groupname username`  

`sudo username -a oG groupname1 groupname2 username`  

### remove a user from a group  

`sudo gpasswd -d username groupname`  

### create a group  

`sudo groupadd groupname`  

### delete a group  

`sudo groupdel groupname`  

### change a user's primary group  

To change a user primary group, use the usermod command followed by the -g option.In the following example, we are changing the primary group of the user linuxize to developers:  

`sudo usermod -g developers  username`  

### create a new user and assign groups in one command  

`sudo useradd -g users -G wheel,developers nathan`  

### display user groups  

`id username`  

### display the user's supplementary groups  

`groups liuwei`  

### sudo -i  

## shell

`touch {1,2}.py`  

### find 
find a diretory named src in the current path  

`find . -name src -type d`  

find a file which path is '\*\*/test/\*.py'  

`find . -path '**/test/*.py' -type f`  

find all files and execute 'rm...'  

`find . -name "*.tmp" -exec rm {} \`  

### 2>&1
有时候我们常看到类似这样的脚本调用：  

`./test.sh  > log.txt 2>&1`  
这里的`2>&1`是什么意思？该如何理解？  
先说结论：上面的调用表明将`./test.sh`的输出重定向到`log.txt`文件中，同时将标准错误也重定向到`log.txt`文件中。  

有何妙用  
我们来看下面的例子，假如有脚本`test.sh:`  

`#!/bin/bash`  
`date         #打印当前时间`  
`while true   #死循环`  
`do`  
`    #每隔2秒打印一次`  
`    sleep 2`  
`    whatthis    #不存在的命令`  
`    echo -e "std output"`  
`done`  

由于系统中不存在`whatthis`命令，因此执行会报错。假如我们想保存该脚本的打印结果，只需将`test.sh`的结果重定向到`log.txt`中即可：  

`./test.sh > log.txt`  
执行结果如下：  

`ubuntu$ ./test.sh >log.txt`
`./test.sh: 行 7: whatthis: 未找到命令`  
我们明明将打印内容重定向到`log.txt`中了，但是这条错误信息却没有重定向到`log.txt`中。如果你是使用程序调用该脚本，当查看脚本日志的时候，将会完全看不到这条错误信息。而使用下面的方式则会将出错信息也重定向到`log.txt`中：  

`./test.sh  > log.txt 2>&1`  

如何理解  
每个程序在运行后，都会至少打开三个文件描述符，分别是`0:stdin;1:stdout;2:stderr.`  
例如，对于前面的`test.sh`脚本，我们通过下面的步骤看到它至少打开了三个文件描述符：  

`./test.sh    #运行脚本`  
`ps -ef|grep test.sh`  #重新打开命令串口，使用`ps`命令找到`test.sh`的`pid`  
`hyb       5270  4514  0 19:20 pts/7    00:00:00 /bin/bash ./test.sh`  
`hyb       5315  5282  0 19:20 pts/11   00:00:00 grep --color=auto test.sh`  
可以看到`test.sh`的`pid`为`5270`，进入到相关`fd`目录：  

`cd /proc/5270/fd   #进程5270所有打开的文件描述符信息都在此`  
`ls -l              #列出目录下的内容`  
`0 -> /dev/pts/7`  
` 1 -> /dev/pts/7`  
` 2 -> /dev/pts/7`  
` 255 -> /home/hyb/workspaces/shell/test.sh`  
可以看到，`test.sh`打开了`0，1，2`三个文件描述符。同样的，如果有兴趣，也可以查看其他运行进程的文件描述符打开情况，除非关闭了否则都会有这三个文件描述符

那么现在就容易理解前面的疑问了，`2>&1`表明将文件描述`2(stderr)`的内容重定向到文件描述符`1(stdout)`，为什么1前面需要`&`？当没有`&`时，`1`会被认为是一个普通的文件，有`&`表示重定向的目标不是一个文件，而是一个文件描述符。在前面我们知道，`test.sh >log.txt`又将文件描述符1的内容重定向到了文件log.txt，那么最终标准错误也会重定向到log.txt。我们同样通过前面的方法，可以看到test.sh进程的文件描述符情况如下：

 `0 -> /dev/pts/7`  
` 1 -> /home/hyb/workspaces/shell/log.txt`  
` 2 -> /home/hyb/workspaces/shell/log.txt`  
` 255 -> /home/hyb/workspaces/shell/test.sh`  
我们可以很明显地看到，文件描述符1和2都指向了`log.txt`文件，也就得到了我们最终想要的效果：将标准错误输出重定向到文件中。
它们还有两种等价写法：

`./test.sh  >& log.txt`  
`./test.sh  &> log.txt`  

### nohup
Nohup is short for “No Hangups.” It’s not a command that you run by itself. Nohup is a supplemental(补充) command that tells the Linux system not to stop another command once it has started. That means it’ll keep running until it’s done, even if the user that started it logs out. The syntax for nohup is simple and looks something like this:

`nohup mycommand`   

`nohup command &`  
eg:`nohup sh your-script.sh &`  

Notice the "&" at the end of the command. That moves the command to the background, freeing up the terminal that you’re working in.  
`nohup ./some-script.sh > ~/Documents/custom.out &`: nohup logs everything to an output file, nohup.out.  

When using &, you'll see the bash job ID in brackets, and the process ID (PID) listed after. For example:
[1] 25132  

You can use the PID to terminate the process prematurely. For instance, to send it the TERM (terminate) signal with the kill command:  

`kill -9 25132`  

### ln -s:create symbolic link  
cd 到你想要建立软连接的目录，直接 ln -s 原文件的路径，创建软链接；假如要改名字的话， ln -s 原文件路径 软链接名字


