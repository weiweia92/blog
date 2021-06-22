## conda
### 1.Conda
**Package Manager+Environment Manager**  
conda包和环境管理器包含于Anaconda的所有版本当中。  
Package Manager：当安装一个包的时候，依赖包  
Environment Manager：create multiple virtual environment  
conda为Python项目而创造，但可适用于上述的多种语言。  
```
conda update conda # 更新conda
conda update anaconda #更新anaconda
conda update python  #
conda list #查看当前环境所有包
conda list -n <env_name> #查看某个环境已安装包
conda search <package_name> # 搜索package信息
conda update -n <env_name> <package_name> #更新指定环境的某个包
conda remove -n <env_name> <package_name> #删除指定环境的某个包
```
5.安装package到指定的环境：conda install -n py35 numpy # 如果不用-n指定环境名称，则被安装在当前活跃环境,也可以通过-c指定通过某个channel安装  
![]()  
#### Environment
```
conda --version #查看conda版本
conda info -e  #查看当前系统中已经安装了哪些环境
conda env list #查看当前系统中已经安装了哪些环境
```
```
conda env create -f environment.yml  #通过yml文件创建新的conda 环境
```
```
source activate ai37 #切换到名字为ai37的环境下
source deactivate ai37  #返回到主环境
conda remove --name ai37 --all  #删除名为ai37的环境
```
#### Package
1. 安装包conda install <package_name>
2. 查看当前环境所有包  
```
conda list  #查看当前环境所有包
conda install <package_name>  #安装包
```
#### Channels
channels:the path or the location where conda looks for the packages that we want to install.  
![]()  
1. 查看当前channel:conda config --show channels
2. 安装特定channel下的包:conda install -c pytorch pytorch (此pytorch channel仅使用了一次，并没有加入到channel list中) 
3. condaforge.com :A community-led collection of recipes, build infrastructure and distributions for the conda package manager
conda config --add channels conda-forge
4. 往默认channel中填加新的channel:`conda config -add channels pytorch`
`conda install pytorch`
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`  
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`  
`conda config --set show_channel_urls yes`  
We deactivate the environment,not the channel
### Anaconda
Anaconda是一个包含180+的科学包及其依赖项的发行版本。其包含的科学包包括：conda, numpy, scipy, ipython notebook等  
Anaconda->conda+环境管理器+1000+开源库  
### pip
pip是用于安装和管理软件包的包管理器。pip编写语言：Python。
Python中默认安装的版本：Python 2.7.9及后续版本默认安装命令为 pip,Python 3.4及后续版本默认安装命令为 pip3  
### pip vs conda
#### 1.依赖项检查
pip：  
不一定会展示所需其他依赖包  
安装包时或许会直接忽略依赖项而安装，仅在结果中提示错误。  
conda：  
列出所需其他依赖包。  
安装包时自动安装其依赖项。  
可以便捷地在包的不同版本中自由切换。  
#### 2.环境管理
pip：维护多个环境难度较大  
conda：比较方便地在不同环境之间进行切换，环境管理较为简单  
#### 3.对系统自带Python的影响  
pip：在系统自带Python中包的更新/回退版本/卸载将影响其他程序    
conda：不会影响系统自带Python    
#### 4.适用语言
pip：仅适用于Python  
conda：适用于Python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN  
conda结合了pip和virtualenv的功能  
