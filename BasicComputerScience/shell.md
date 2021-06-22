## Shell
### 常见Linux目录
1. /bin : 二进制目录，存放很多用户级的GNU工具
2. /boot : 启动目录，存放启动文件
3. /dev : 设备(device)目录，linux在这里创建设备节点
4. /etc : 系统配置文件目录
5. /mnt : 挂载目录
6. /opt : 可选目录，常用于存放第三方软件包和数据文件
7. /proc : 进程目录，存放现有硬件及当前进程的相关信息
8. /sbin : 系统二进制目录，存放许多GNU管理员级工具
9. /usr : 用户二进制目录  
10. /var : 可变目录
11. /srv : 服务目录，存放本地服务相关文件
### 环境变量
```
# 数组变量
(base) admin@try:/mnt/weiweia92$ mytest=(one two three four five)
(base) admin@try:/mnt/weiweia92$ echo $mytest
one
(base) admin@try:/mnt/weiweia92$ echo ${mytest[2]}
three
(base) admin@try:/mnt/weiweia92$ echo ${mytest}
one
(base) admin@try:/mnt/weiweia92$ echo ${mytest[*]}
one two three four five
(base) admin@try:/mnt/weiweia92$ mytest[2]=seven   #改变数组变量的某个值
(base) admin@try:/mnt/weiweia92$ echo ${mytest[*]}
one two seven four five
(base) admin@try:/mnt/weiweia92$ unset mytest[2]   #删除数组变量中的某个值
(base) admin@try:/mnt/weiweia92$ echo ${mytest[*]}
one two four five
(base) admin@try:/mnt/weiweia92$ echo ${mytest[2]}

(base) admin@try:/mnt/weiweia92$ echo ${mytest[3]}
four
(base) admin@try:/mnt/weiweia92$ unset mytest     #删除整个数组变量
(base) admin@try:/mnt/weiweia92$ echo ${mytest[*]}
```
### Redirection
#### output redirection
```
(base) admin@try:/mnt/weiweia92$ date > test6 # >:若test6文件里有内容则将其覆盖
(base) admin@try:/mnt/weiweia92$ cat test6
Tue Oct 13 14:53:51 CST 2020
(base) admin@try:/mnt/weiweia92$ date >> test6  # >>:不覆盖,追加到已有文件中
(base) admin@try:/mnt/weiweia92$ cat test6
Tue Oct 13 14:53:51 CST 2020
Tue Oct 13 14:54:47 CST 2020
```
#### input redirection  
```
(base) admin@try:/mnt/weiweia92$ wc < test6
 2 12 58
 # wc: 文本行数 文本次数 文本字节数
```
#### inline input redirection
```
# 格式：
#command << marker
#>...
#>...
#>marker

(base) admin@try:/mnt/weiweia92$ wc << EOF
> test string1
> test string2
> test string3
> EOF
 3  6 39
```
### Shell Script
#### Compare
Numerical comparison  
```
n1 -eq n2 # equal
n1 -ge n2 # greater equal
n1 -gt n2 # greater
n1 -le n2 # less that or equal
n1 -lt n2 # less that
n1 -ne n2 # not equal
```
String comparison  
```
str1 = str2
str1 != str2
str1 \< str2
str1 \> str2
-n str1 # check the length of string is not 0
-z str1 # check the length of string is 0
```
File comparison  
```
-d file # check whether or not exists and it is a directory
-e file # check whether or not exists
-f file # check whether or not exists and it is a file
-r file # check whether or not exists and readable
-s file # check whether or not exists and it is not null
-w file # check whether or not exists and it is writable
-x file # check whether or not exists and it is executable
file1 -nt file2 # whether or not file1 is newer than file2
file1 -ot file2 # whether or not file1 is older than file2
```
#### Functions
```
#defining and using a function
#!/bin/bash

myfunc() {
				echo "Using function"
}
total=1
while [ $total -le 3 ]; do
			myfunc
      total=$(($total + 1))
done
echo "Loop finished"
myfunc
echo "End of the script"
```
![]()  
```
#!/bin/bash
myfunc() {
				read -p "Enter a value: " value
        echo $(($value + 10))
}
result=$(myfunc)
echo "The value is $result"
```
![]()  
### shopt
Linux shell有交互式和非交互式两种工作模式。日常使用shell输出命令得到结果的方式是交互式的方式，而shell脚本使用的是非交互式的  
```
### shopt
shopt -s opt_name                 #Enable (set) opt_name.
shopt -u opt_name                 #Disable (unset) opt_name.
shopt opt_name                    #Show current status of opt_name
```
![]()  
从上图可以看出交互式模式alias扩展功能是开启的  
```
#!/bin/bash

alias echo_hello='echo hello!'
shopt expand_aliases
echo_hello

shopt -s expand_aliases
shopt expand_aliases
echo_hello
```
![]()  
可以看到在非交互的情况下默认是关闭的但是我们可以用shopt来将其开启。  
另外，alias别名只在当前shell有效，不能被子shell继承，也不能像环境变量一样export。可以把alias别名定义写在.bashrc文件中，这样如果启动交互式的子shell，则子shell会读取.bashrc，从而得到alias别名定义。但是执行shell脚本时，启动的子shell处于非交互式模式，是不会读取.bashrc的。  
不过，如果你一定要让执行shell脚本的子shell读取.bashrc的话，可以给shell脚本第一行的解释器加上参数：  
#!/bin/bash --login  
–login使得执行脚本的子shell成为一个login shell，login shell会读取系统和用户的profile及rc文件，因此用户自定义的.bashrc文件中的内容将在执行脚本的子shell中生效。  
还有一个简单的办法让执行脚本的shell读取.bashrc，在脚本中主动source ~/.bashrc即可。  
### Special symbol
```
$#      添加到shell的参数个数
$0      shell本身的文件名
$1 ~ $n 添加到shell的各个参数值 $1表示第一参数，$n表示第n个参数
$?      上一个命令执行后的退出状态
$!      最后执行的后台命令的PID
$$      所在命令的PID
$_      上一个命令的最后一个参数
$*      以一对双引号给出参数列表
$@      将各个参数分别加双引号返回
```
### Command line
#### ls
```
# ls支持命令行中定义过滤器，过滤器实际就是进行简单文本匹配的字符串，可以添加到命令行参数之后
ls -l my_script  # 列出my_script文件的信息
ls -l my_scr?pt  # ?代表一个字符
ls -l my_scr*    # *代表零个或多个字符
ls -l my_scr[ai]pt # 输出my_scrapt 和my_script
ls -l my_scr[a-i]pt
ls -l f[!a]ll 
```
#### cat
```
(base) admin@try:/mnt/weiweia92$ cat -n test1  # -n:所有行加行号
     1  hello
     2
     3  This is a test file.
     4
     5
     6  That we'll use to       test the cat command.
(base) admin@try:/mnt/weiweia92$ cat -b test1 #-b: 只给有文本的行加行号
     1  hello

     2  This is a test file.


     3  That we'll use to       test the cat command.
(base) admin@try:/mnt/weiweia92$ cat -T test1 # -T:不让制表符(tab)出现
hello

This is a test file.


That we'll use to ^Itest the cat command.
(base) admin@try:/mnt/weiweia92$ cat test1 
hello

This is a test file.


That we'll use to       test the cat command.
```
#### less
#### tail/head
```
tail -n 2 log_file #显示log_file的后两行（tail默认为10行）
head -5 log_file #显示log_file的前五行
```
#### mkdir and rmdir
```
odir=$outdir/$ts/
mkdir -p $odir #若$ourdir不存在，则建立一个(若不加-p,且$outdir不存在，则会报错)
rmdir $odir    #删除$odir
```
#### chown and chmod
The mode which consists of 3 parts,owner,group and others.   
Read=4, write=2, execute=1  
```
chmod 755 myfile
# owner:7=4+2+1, group:5=4+1, other:5=4+1
# Note:execute for a folder,means opening it.
```
#### tar
```
tar -czvf myfiles.tar.gz myfiles #压缩myfiles为myfiles.tar.gz
tar -xzvf mytar.tar.gz #解压
```
#### file  
```
file myfile # viewing the file type
```
#### du/df
```
du -hs . #当前各文件大小
df -h    # show the disk free space
```
#### exit
`Linux exit`命令用于退出目前的`shell`。   
执行`exit`可使`shell`以指定的状态值退出。若不设置状态值参数，则`shell`以预设值退出。状态值0代表执行成功，其他值代表执行失败。`exit`也可用在`script`，离开正在执行的`script`，回到`shell`   
#### sed (The tool of text processing)
用于字符串操作的便捷工具。
```
# 作用范围在全文
sed 's/cat/dog/g' pet.txt

# 替换全文每一行的第1个 cat 为 dog
sed 's/cat/dog/' pet.txt

# 作用范围在第1行
sed '1s/cat/dog/g' pet.txt

# 作用范围在第6行到第10行
sed '6,10s/cat/dog/g' pet.txt

# 作用范围在第6行到最后一行
sed '6,$s/cat/dog/g' pet.txt

# 作用范围在指定行到其后2行，用加号(减号不可用)
sed '1,+2s/cat/dog/g' pet.txt

# 替换第1行一整行为dog
sed '1s/.*/dog/' pet.txt

# 这里因为.*已代表一整行，所有后面写上/g和上面相同效果
sed '1s/.*/dog/g' pet.txt

# 替换全文的每1行为dog
sed 's/.*/dog/' pet.txt

# 这里因为.*已代表一整行，所有后面写上/g和上面相同效果
sed 's/.*/dog/g' pet.txt

# 替换第1行的每一个字符
sed '1s/./dog/g' pet.txt

# 替换第1行的第1个字符
sed '1s/./dog/' pet.txt

# 替换第1行的第5个字符
sed '1s/./dog/5' pet.txt

$ echo '123'|sed 's/./dog/g'
dogdogdog
$ echo '123'|sed 's/./dog/'
dog23
$ echo '123'|sed 's/./dog/3'
12dog
# 字符也包括符号
$ echo ',123'|sed 's/./dog/1'
dog123

```
* 不打印出全文，仅打印更改所涉及行，或者说仅打印受影响的行在sed后面加-n,是阻止默认的自动打印模式的选项，同时在 替换目标option 的位置 写上p，表明打印print。
```
# 打印发生替换的行
sed -n 's/cat/dog/gp' pet.txt
```
#### sed的y命令
不同于上面的s命令，以字符串或模式为单位替换为一个整体，y是罗列置换每个对应的字符。
语法：  
`sed 'y/查找的各个字符/对应替换后的各个字符/' 文件名`  
`sed ‘y/abc/123’ test.txt`，这个命令会依次替换a，b，c为1，2，3 。查找的各个字符与对应替换后的各个字符的长度要一致。  
```
$ echo 'a,b,c,d,e'|sed 'y/abcde/12345/'
1,2,3,4,5

$ cat test.txt
a,b,c

$ sed 'y/abcde/12345/' test.txt
1,2,3

$ cat copy.txt 
wang yi
zhang san
li qi

# 想要把1-2行的小写转化为大写，正则表达式不可用
$ sed '1,2y/[a-z]/[A-Z]/' copy.txt 
wAng yi
ZhAng sAn
li qi

# 罗列全部字母，来替换
$ sed '1,2y/abcdefghijklmnopqrstuvwxyz/ABCDEFGHIJKLMNOPQRSTUVWXYZ/' copy.txt 
WANG YI
ZHANG SAN
li qi
```
#### awk
#### gsub
gsub函数使得所有正则表达式被匹配时都发生替换   
gsub(regular expression, sub_str, target_str)   
```
admin@weiwei:/$ echo "a b c 2011-11-22 a:d" | awk '$4=gsub(/-/,"",$4)'  #gsub返回的是替换的次数
a b c 2 a:d
admin@weiwei:/mnt$ echo "a b c 2011-11-22 a:d" | awk 'gsub(/-/,"_",$4)'
a b c 2011_11_22 a:d
```
```
#有一个文本文件data.test的内容如下：
#将数字去掉其他不变

0001|20081223efskjfdj|EREADFASDLKJCV
0002|20081208djfksdaa|JDKFJALSDJFsddf

解法：
awk -F '|' 'BEGIN{ OFS="|" } {sub(/[0-9]+/,"",$2);print $0}' data.test
or
awk -F '|' -v OFS='|' '{sub(/[0-9]+/,"",$2);print $0}' data.test
```
#### seq
```
(base) admin@try:/$ seq -s , 1 9 #打印从1到9的序列并以,分隔
1,2,3,4,5,6,7,8,9
```
```
(base) admin@try:/$ seq -f "%04g" 2 4
0002
0003
0004
```
#### pushd and popd
pushd和popd使用栈的方式来管理目录  
pushd:将目录加入到栈顶部，并将当前目录切换到该目录。若不加任何参数，该命令用于将栈顶的两个目录进行对调  
popd:删除目录栈中的目录。若不加任何参数，则会首先删除目录栈顶的目录，并将当前目录切换到栈顶下面的目录。  
#### rsync  
```
sudo apt install rsync
rsync -a /path/to/source/ /path/to/destination #将多个文件夹中的文件合并到一个文件夹
```
#### iconv
#### rename
```
rename 's/^/logo/' *.png  #将*.png变为logo*.png 即对文件加前缀
rename -d logo *.png  #删除前缀logo
```
### For loop
```
#!/bin/bash
for var in first "second" third fourth "I'll do it"; do
echo This is: $var item
done
```
```
(base) admin@try:/mnt/weiweia92/pai_speech_synthesis/learn_bash$ bash 3.sh
This is: first item
This is: second item
This is: third item
This is: fourth item
This is: I'll do it item
```