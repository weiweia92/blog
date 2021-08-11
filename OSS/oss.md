# 基本概念

## 1. bucket(存储空间)

存储空间是用户用于object(存储对象)的容器，所有对象都隶属于某个存储空间。存储空间具有各种配置属性，包括地域、访问权限、存储类型等。

* 同一个存储空间的内部是扁平的，没有文件系统的目录等概念，所有的对象都直接隶属于其对应的存储空间。
* 每个用户可以拥有**多个存储空间**。同一阿里云账号在同一地域内创建的存储空间总数不能超过 100 个
* 存储空间的名称在OSS范围内必须是全局唯一的，一旦创建之后无法修改名称。
* 存储空间内部的对象数目没有限制。
* 存储空间一旦创建成功，名称和所处地域不能修改。

### 1.1 创建bucket
#### ossutil

命令行：   
`./ossutil64 mb oss://bucketname `     

例如：
`./ossutil64 mb oss://examplebucket -e oss-cn-shanghai.aliyuncs.com -i LTAI4Fw2NbDUCV8zYUzA****  -k 67DLVBkH7EamOjy2W5RVAHUY9H****`    

具体参数使用：[参数说明](https://help.aliyun.com/document_detail/50455.htm?spm=a2c4g.11186623.2.7.6f6c6b81q8wKRj#section-yhn-ko6-gqj)

#### python SDK

`bucket.create_bucket()`

设置权限的参数：   
oss2.BUCKET_ACL_PRIVATE:私有    
oss2.BUCKET_ACL_PUBLIC_READ:公共读     
oss2.BUCKET_ACL_PUBLIC_READ_WRITE:公共读写


举例： 
```
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

# 如果需要在创建存储空间时设置存储类型、存储空间访问权限、数据容灾类型，请参考以下代码。
# 以下以配置存储空间为标准存储类型，访问权限为私有，数据容灾类型为同城冗余存储为例。
bucketConfig = oss2.models.BucketCreateConfig(oss2.BUCKET_STORAGE_CLASS_STANDARD, oss2.BUCKET_DATA_REDUNDANCY_TYPE_ZRS)
# 设置存储空间的存储类型为低频访问类型，访问权限为公共读。
#bucket.create_bucket(oss2.BUCKET_ACL_PUBLIC_READ, oss2.models.BucketCreateConfig(oss2.BUCKET_STORAGE_CLASS_IA)) 
bucket.create_bucket(oss2.BUCKET_ACL_PRIVATE, bucketConfig)     
```
```
# -*- coding: utf-8 -*-
import oss2

# 阿里云主账号AccessKey拥有所有API的访问权限，风险很高。强烈建议您创建并使用RAM账号进行API访问或日常运维，请登录 https://ram.console.aliyun.com 创建RAM账号。
auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

bucket.create_bucket() # 设置为默认
```
```
print(bucket.get_bucket_acl().acl)     # 获取bucket的访问权限
```

### 1.2 列举bucket

#### ossutil

命令行：

`./ossutil64 ls` 或者 `./ossutil64 ls oss://`

[详细使用](https://help.aliyun.com/document_detail/120052.htm?spm=a2c4g.11186623.2.7.40a84955XRU4Ga#section-qz8-3f3-3pp)

### python SDK

```
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

# 获取存储空间的地域信息。
result = bucket.get_bucket_location()
print('location: ' + result.location)     
```
### 1.3 判断bucket是否存在

#### python SDK

```
def does_bucket_exist(bucket):
    try:
        bucket.get_bucket_info()
    except oss2.exceptions.NoSuchBucket:
        return False
    except:
        raise
    return True

exist = does_bucket_exist(bucket)

if exist:
    print('bucket exist')
else:
    print('bucket not exist')
```
### object(对象)

object是oss存储数据的基本单元，也称为oss的文件。和传统的文件系统不同，object没有文件目录层级结构的关系。object由object meta(元信息),data(用户数据)和key(文件名)组成，并由存储空间内唯一的key来标识。object meta是一组键值对，表示object的一些属性，如最后修改时间，大小等信息，同时用户也可以在object meta中存储一些自定义的信息。

object的生命周期是从上传成功到被删除为止。在整个生命周期中，除通过追加方式上传的object可以通过继续追加上传写入数据外，其他方式上传的object内容无法编辑，可以通过重复上传同名object来覆盖之前的Object.

### objectkey

objectkey, key以及objectname是同一概念，均表示对object执行相关操作时需要填写的object名称。例如向某一存储空间上传object时，objectKey表示上传的object所在存储空间的完整名称，即包含文件后缀在内的*完整路径*，如填写为abc/efg/123.jpg。

### Region(地域)

Region表示OSS的数据中心所在物理位置。         
eg:     
Region:华东1（杭州）           
Region ID:oss-cn-hangzhou          
Endpoint:oss-cn-hangzhou-internal.aliyuncs.com    
