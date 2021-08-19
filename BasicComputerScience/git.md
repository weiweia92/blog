## Git

Git-version control system to locally track changes in your project/folder and push & pull changes from remote repositories like GitHub, BitBucket,GitLab

## GitLab,BitBucket,GitHub

Services that allow to host your project on a remote repo & have additional features to help in SDLC(software development lifecycle) and CI(continuous integration),CD(continuous delivery).
eg: Managing  Sharing   Wiki  Bug tracking CI & CD

### 1.GIT设置

```
git config --global user.name "weiweia92"
git config --global user.name 
git config --global user.email "weiweia92@163.com"
git config --global user.email
git config --global --list #it show your username,email and password, it can check all
#your settings and all the values that you have set
ssh-keygen -t rsa -C "weiweia92@163.com"  #生成ssh key
```
### 2.WorkSpace->Stage->Local Repo->Remote Repo

![](https://github.com/weiweia92/blog/blob/main/BasicComputerScience/pic/1615359094257-aa9f97e9-c080-4950-bcd8-c6b2e2c926ce.png)
工作区(workspace)：工作区就是你克隆项目到本地后，项目所在的文件夹目录。

暂存区(Index/stage)：用于存储工作区中添加上来的变更（新增、修改、删除）的文件的地方。操作时，使用git add .会将本地所有新增、变更、删除过的文件的情况存入暂存区中。

本地仓库(Local Repo)：用于存储本地工作区和暂存区提交上来的变更（新增、修改、删除）过的文件的地方。操作时，使用git commit –m “本次操作描述” 可以将添加到暂存区的修改的文件提交到本地仓库中。

远程仓库(Remote Repo)：简单来说，就是我们工作过程中，当某一个人的开发工作完毕时，需要将自己开发的功能合并到主项目中去，但因为功能是多人开发，如果不能妥善保管好主项目中存储的代码及文件的话，
将会存在丢失等情况出现，所以不能将主项目放到某一个人的本地电脑上，这时就需要有一个地方存储主项目，这个地方就是我们搭建在服务器上的git远程仓库，也就是在功能开始开发前，每个人要下载项目到本地的地方。
操作时，使用git push origin 分支名称，将本次仓库存储的当前分支的修改推送至远程仓库中的对应分支中。

```
git init #在本地的当前目录初始化git仓库
git diff #显示WorkSpace和Stage中的状态差异
git add <file> / git add .  #从WorkSpace保存到Stage,add后的文件才会被git跟踪
git mv <old> <new> #文件改名
git rm <file> #删除文件
git rm --cached <file> #从Stage中移除，停止跟踪文件但不删除
git commit -m "message" #从Stage提交更新到Local Repo,但还没有到远程仓库中

####如果本地仓库的变更也已经保存到了远程仓库
git push -f #强制推送到远程分支

git diff <source_brach> <target_branch>  #对比两分支差异

git commit -a -m "msg4" #git add . + git commit -m "msg4"
#在push到远端之前突然想到还有其他修改，将这个修改file4.txt合并到msg4的commit里，用--amend
git add file4.txt
git commit --amend
```

```
git checkout --<file> #丢失file自从上次commit后的所有改动
git reset HEAD <file> #撤销add,用于反悔git add <file>--绿字变红字 未跟踪状态
####如果此时突然不想修改本地仓库了，即将本地仓库回滚到之前的版本
git reflog  #查看提交变更之前的版本号，由英文字母和数字组合的字符串<commit ID>
git reflog -n #指定显示条数
git reset --hard 45a992bc53bd4 #45a992bc53bd4：<commit ID> 将本地仓库回滚到指定的版本
```

![](https://github.com/weiweia92/blog/blob/main/BasicComputerScience/pic/1615280056033-d91d8410-5cc8-4b04-92f0-3407141797e8.png)

HEAD值越小，表示版本越新，越大表示版本生成时间越久。    
在上面图中，我们发现HEAD值的展示形式为HEAD@{0}、 HEAD@{1}、HEAD@{2}...同样HEAD值的数字越小，表示版本越新，数字越大表示版本越旧。     
如果你还没有clone现有仓库，并欲将你的仓库连接到某个远端服务器：   

```
git remote add origin <server> # 如此你就能够将你的改动推送到所添加的服务器上去了
git remote -v  #显示抓取和推送地址

通过git reset --hard 版本号 即可将本地仓库回滚到指定的版本，如果本地的变更也已经保存到了远程仓库，
#我们此时可以再输入git push -f将本地仓库回滚后的版本强制提交到远程分支
```
如果我们通过git commit -m “注释” 提交变更到了本地仓库，但是突然不想修改本仓库了，我们可以通过git reflog查看提交变更之前的版本号，由英文字母和数字组合的字符串，获取之前的版本号之后，
通过git reset --hard 版本号 即可将本地仓库回滚到指定的版本，如果本地的变更也已经保存到了远程仓库，我们此时可以再输入git push -f将本地仓库回滚后的版本强制提交到远程分支     

```
git log #查看commit的历史记录（当前分支）
git log -p <file>  #查看指定文件的提交历史
git blame <file>  #以列表方式查看指定文件的提交历史
```
![](https://github.com/weiweia92/blog/blob/main/BasicComputerScience/pic/1609157251147-5ab7db61-30db-4e23-a141-72323bd193f0.png)
```
git log --oneline #查看当前所在分支/对应远程分支 每次提交的版本号
```
![](https://github.com/weiweia92/blog/blob/main/BasicComputerScience/pic/1609157407912-84d94229-75cb-45f2-aa54-908edd9e72e4.png)   

### 2.BRANCH

#### 2.1.local branch

branch是用来将特性开发绝缘开来的。在你创建仓库时，master是默认的branch，在其他分支开发，完成后再将它们合并到主分支上。    
![](https://github.com/weiweia92/blog/blob/main/BasicComputerScience/pic/1609148476778-92ba099b-b880-4536-8124-7a37abf78ded.png)    
```
git branch  #查看本地分支
git branch -r  #查看远程分支
git branch -a  #查看所有分支
git branch -d <local_branch_name>  #删除某一本地分支
git push origin --delete <local_branch_name> #将删除的本地分支同步到远端
git checkout <branch/tag>  #切换分支(本地)或标签
#创建一个branch_name分支并切换过去，除非你将该分支推送到远端仓库，不然该分支就是不为他人所见的
git checkout -b <branch_name> 
git tag #列出所有本地标签
git tag <tagname> #基于最新提交创建标签
git tag -d <tagname> #删除标签

#将本地建立的pre_merge_model与远程的pre_merge_model进行关联
git branch --set-upstream-to=origin/pre_merge_model pre_merge_model  

#要将本地更改提交到远程分支上之前需要pull一下远程分支
git pull
```
Note:不能删除当前所在的分支，要切换到master分支后再删除该分支。      

1.创建week1分支`git checkout -b week1`  

2.在week1分支下进行更改，然后将其更改commit,这个commit会同步到远端的这个分支上        

`git commit -a -m "msg5"`        

3.如果远端没有这个分支则使用`git push -u origin week1` 即可. 

#### 2.2.remote

```
git remote -v #查看远程版本库信息
#将创建好的本地分支push到远程仓库 <branch name>:要提交到远程仓库的本地分支名
git push origin <branch name>   
#删除某一远程分支
git push origin --delete <origin_branch_name> 
```
#### 2.3.gitlab给分支设定权限

给gitlab的各位开发设置权限是很重要的，不然他们就可能会偷偷把执行分支合并甚至git pull来破坏线上环境。
project--setting--members    

![](https://github.com/weiweia92/blog/blob/main/BasicComputerScience/pic/1609147666699-82157a9a-c1f5-45fc-967a-c9198f76e5b3.png)     
Master,Developer,Reporter,Reporter只有读权限，可以创建代码片段，一般来说给测试人员，Guest只能提交问题和评论。   
```
#拉取远程pre_merge分支到本地pre_merge(自己命名)的分支
git checkout -b pre_merge origin/pre_merge
```
### 3.MERGE

将week1分支合并到master分支上
```
git checkout week1
git commit -a -m "msg6"
git checkout master
git merge week1
#在merge时可能会有冲突，根据提示手动更改冲突，之后
git add file4.txt
git commit -m "msg7"
```

**git 远程分支已经把某个分支合并到master中，但是本地通过git branch -r依然可以看到远程分支里有该分支，解决方法：**

```
git remote show origin 
#查看remote地址，远程分支，还有远程分支与之相应的信息，我们可以看到那些远程分支已经不存在的分支
git remote prune origin
#此时再查看远程分支可以发现已经删除了那些远程仓库不存在的分支
```
