### 1. cat(输出文件内容)

将未开启版本控制的目标存储空间examplebucket内名为test.txt的文件内容输出到屏幕。   

命令格式:    
`./ossutil64 cat oss://bucketname/objectname --payer <value> --version-id <value>`     

参数:    
--payer:请求的支付方式。如果希望访问指定路径下的资源产生的流量、请求次数等费用由请求者支付，请将此选项的值设置为requester。    
--version_id:Object的指定版本。仅适用于已开启或暂停版本控制状态Bucket下的Object。

```
./ossutil64 cat oss://examplebucket/test.txt
```

将已开启版本控制的目标存储空间examplebucket内名为exampleobject.txt文件的指定版本内容输出到屏幕.     

```
./ossutil64 cat oss://examplebucket/exampleobject.txt --version-id  CAEQARiBgID8rumR2hYiIGUyOTAyZGY2MzU5MjQ5ZjlhYzQzZjNlYTAyZDE3****
```

例如您需要查看另一个阿里云账号下，华东2（上海）地域下目标存储空间examplebucket1下名为exampleobject1.txt文件内容，命令如下：     

```
./ossutil64 cat oss://examplebucket1/exampleobject1.txt -e oss-cn-shanghai.aliyuncs.com -i LTAI4Fw2NbDUCV8zYUzA****  -k 67DLVBkH7EamOjy2W5RVAHUY9H****
```
### 2. cp

#### 2.1 上传文件

* 本地文件：examplefile.txt（根目录下的文件）
* 本地文件夹：localfolder（根目录下的文件夹）
* 目标Bucket：examplebucket
* 目标Bucket指定目录：desfolder

**上传单个文件**     

```
./ossutil cp examplefile.txt oss://examplebucket/desfolder/
```

** 仅上传文件夹内的文件    **

```
./ossutil cp -r localfolder/ oss://examplebucket/desfolder/
```
 
**上传文件夹及文件夹内的文件     **

```
./ossutil cp -r localfolder/ oss://examplebucket/desfolder/localfolder/
```

**上传文件夹并跳过已有文件**

批量上传失败重传时，可以指定--update（可缩写为-u）选项跳过已经上传成功的文件，实现增量上传。  

```
./ossutil cp -r localfolder/ oss://examplebucket/desfolder/ -u
```

**仅上传当前目录下的文件，忽略子目录      **

```
./ossutil cp localfolder/ oss://examplebucket/desfolder/ --only-current-dir -r
```

**批量上传符合条件的文件**

1.上传所有文件格式为txt的文件       
```
./ossutil cp localfolder/ oss://examplebucket/desfolder/ --include "*.txt" -r
```
2.上传所有文件名包含abc且不是jpg和txt格式的文件      
```
./ossutil cp localfolder/ oss://examplebucket/desfolder/ --include "*abc*" --exclude "*.jpg" --exclude "*.txt" -r
```

#### 2.2 下载文件

**下载单个文件**
* 沿用源文件名保存文件.      

```
./ossutil cp oss://examplebucket/desfolder/examplefile.txt localfolder/
```
* 按指定文件名保存文件. 

```
./ossutil cp oss://examplebucket/desfolder/examplefile.txt localfolder/example.txt
```

**批量下载并跳过已有文件**

```
./ossutil cp -r oss://examplebucket/desfolder/  localfolder/  --update                           
```

**仅下载当前目录下的文件，忽略子目录     **

```
./ossutil cp oss://examplebucket/desfolder/ localfolder/ --only-current-dir -r
```

**范围下载     **         

下载文件时，可以通过 --range选项指定下载范围。例如将 examplefile.txt的第10到第20个字符作为一个文件下载到本地      

```
./ossutil cp oss://examplebucket/desfolder/examplefile.txt localfolder/  --range=10-20
```


**批量下载符合指定条件的文件    **

* 下载所有文件格式不为jpg的文件     
  ```
  ./ossutil cp oss://examplebucket/desfolder/ localfolder/ --exclude "*.jpg" -r
  ```
* 下载所有文件名包含abc且不是jpg和txt格式的文件        
  ```
  ./ossutil cp oss://examplebucket/desfolder/ localfolder/ --include "*abc*" --exclude "*.jpg" --exclude "*.txt" -r
  ```

#### 2.3 拷贝文件

**拷贝单个文件**

```
./ossutil cp oss://examplebucket1/examplefile.txt oss://examplebucket1/srcfolder2/                                 
```

**拷贝文件夹    **

```
./ossutil cp oss://examplebucket1/srcfolder1/ oss://examplebucket2/desfolder/ -r                                   
```

**拷贝增量文件** 

批量拷贝时，若指定 --update选项，只有当目标文件不存在，或源文件的最后修改时间晚于目标文件时，ossutil才会执行拷贝操作。命令如下：    
```
./ossutil cp oss://examplebucket1/srcfolder1/ oss://examplebucket2/path2/ -r --update
```

**重命名   **

```
./ossutil cp oss://examplebucket1/examplefile.txt oss://examplebucket1/example.txt                        
```

**仅拷贝当前目录下文件，忽略子目录    **

```
./ossutil cp oss://examplebucket1/srcfolder1/ oss://examplebucket1/srcfolder2/ --only-current-dir -r
```

### 3. create-symlink

**为目标存储空间examplebucket根目录下的test.jpg文件创建名为example.jpg的软链接文件，并将软链接文件保存至该Bucket下的destfolder目录。    **

```
./ossutil64 create-symlink  oss://examplebucket/destfolder/example.jpg  oss://examplebucket/test.jpg 
```
**例如您需要为另一个阿里云账号下，华东2（上海）地域下目标存储空间testbucket下的exampleobject.png文件创建名为testobject.png的软链接.     **

```
./ossutil64 create-symlink  oss://testbucket/testobject.png  oss://testbucket/exampleobject.png -e oss-cn-shanghai.aliyuncs.com -i LTAI4Fw2NbDUCV8zYUzA****  -k 67DLVBkH7EamOjy2W5RVAHUY9H****
```

### 4. du

获取指定bucket,文件目录下包含的所有object的大小。     

**命令用于查询examplebucket内所有版本Object的大小.   **

```
./ossutil64 du oss://examplebucket --all-versions
```

**查询examplebucket内指定目录dir下的当前版本Object大小，Object大小以GB为单位进行统计** 

```
./ossutil64 du oss://examplebucket/dir/  --block-size GB
```

**查询目标存储空间examplebucket下与前缀test匹配的所有版本Object的大小，Object大小以KB为单位进行统计**

```
./ossutil64 du oss://examplebucket/test --all-versions --block-size KB
```















