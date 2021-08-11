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
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

bucket.create_bucket() # 设置为默认
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
service = oss2.Service(auth, 'http://oss-cn-hangzhou.aliyuncs.com')

# 列举所有的存储空间
for b in oss2.BucketIterator(service):
    print(b.name)
    
# 列举前缀为test-的存储空间。
for b in oss2.BucketIterator(service, prefix='test-'):
    print(b.name)
    
# 列举按字典序排列在test-bucket1之后的存储空间。列举结果中不包含名为test-bucket1的存储空间。
for b in oss2.BucketIterator(service, marker='test-bucket1'):
    print(b.name)
```

### 1.3 获取bucket的地域

```
# 获取存储空间的地域信息。
result = bucket.get_bucket_location()
print('location: ' + result.location)
```
### 1.4 判断bucket是否存在

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
### 1.5 获得bucket信息

```
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

# 获取存储空间相关信息，包括存储空间的存储类型、创建日期、访问权限、数据容灾类型等。   
bucket_info = bucket.get_bucket_info()
print('name: ' + bucket_info.name)
print('storage class: ' + bucket_info.storage_class)
print('creation date: ' + bucket_info.creation_date)
print('intranet_endpoint: ' + bucket_info.intranet_endpoint)
print('extranet_endpoint ' + bucket_info.extranet_endpoint)
print('owner: ' + bucket_info.owner.id)
print('grant: ' + bucket_info.acl.grant)
print('data_redundancy_type:' + bucket_info.data_redundancy_type)   
```
### 1.6 管理bucket访问权限

设置bucket权限为私有。  
```
# oss2.BUCKET_ACL_PUBLIC_READ:公共读 
# oss2.BUCKET_ACL_PUBLIC_READ_WRITE:公共读写
bucket.put_bucket_acl(oss.BUCKET_ACL_PRIVATE)
```
获取bucket访问权限.只有bucket的拥有者才能获取bucket的访问权限         
```
print(bucket.get_bucket_acl().acl)
```

### 1.7 删除bucket

```
def delete(bucket):
    try:
        # 删除存储空间。
        bucket.delete_bucket()
    except oss2.exceptions.BucketNotEmpty:
        print('bucket is not empty.')
    except oss2.exceptions.NoSuchBucket:
        print('bucket does not exist')
```

### 1.8 bucket标签

只有Bucket的拥有者及授权子账户才能为Bucket设置用户标签，否则返回403 Forbidden错误，错误码：AccessDenied。    

```
import oss2
from oss2.models import Tagging, TaggingRule

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

# 创建标签规则。
rule = TaggingRule()
rule.add('key1', 'value1')
rule.add('key2', 'value2')

# 创建标签。
tagging = Tagging(rule)
# 设置Bucket标签。
result = bucket.put_bucket_tagging(tagging)
# 查看HTTP返回码。
print('http status:', result.status)

# 获取Bucket标签信息。
result = bucket.get_bucket_tagging()
# 查看获取到的标签规则。
tag_rule = result.tag_set.tagging_rule
print('tag rule:', tag_rule)

# 删除Bucket标签。
result = bucket.delete_bucket_tagging()
# 查看HTTP返回码。
print('http status:', result.status)
```

列举带指定标签的bucket   
```
# -*- coding: utf-8 -*-
import oss2

#创建Server对象。
auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
service = oss2.Service(auth,  'http://oss-cn-hangzhou.aliyuncs.com')

#填充tag-key，tag-value字段到list_buckets接口的params参数中。
params = {}
params['tag-key'] = '<yourTagging_key>'
params['tag-value'] = '<yourTagging_value>'

#列举出带指定标签的Bucket。
result = service.list_buckets(params=params)
#查看列举结果。
for bucket in result.buckets:
    print('result bucket_name:', bucket.name)
```
## 2. object(对象)

object是oss存储数据的基本单元，也称为oss的文件。和传统的文件系统不同，object没有文件目录层级结构的关系。object由object meta(元信息),data(用户数据)和key(文件名)组成，并由存储空间内唯一的key来标识。object meta是一组键值对，表示object的一些属性，如最后修改时间，大小等信息，同时用户也可以在object meta中存储一些自定义的信息。

object的生命周期是从上传成功到被删除为止。在整个生命周期中，除通过追加方式上传的object可以通过继续追加上传写入数据外，其他方式上传的object内容无法编辑，可以通过重复上传同名object来覆盖之前的Object.

### 2.1 简单上传

上传 object 时，如果 bucket 中已存在同名 object 且用户对该 object 有访问权限，则新添加的 object 将覆盖原有 object。   

上传文件    

```
import oss2

auth = oss2.Auth('yourAccessKeyId', 'yourAccessKeySecret')
bucket = oss2.Bucket(auth, 'yourEndpoint', 'examplebucket')

# 上传文件。
# 如果需要在上传文件时设置文件存储类型（x-oss-storage-class）和访问权限（x-oss-object-acl），请在put_object中设置相关Header。
# headers = dict()
# headers["x-oss-storage-class"] = "Standard"
# headers["x-oss-object-acl"] = oss2.OBJECT_ACL_PRIVATE
# 填写Object完整路径和字符串。Object完整路径中不能包含Bucket名称。
# result = bucket.put_object('exampleobject.txt', 'Hello OSS', headers=headers)
result = bucket.put_object('exampleobject.txt', 'Hello OSS') # 上传的字符串为'Hello OSS'

# HTTP返回码。
print('http status: {0}'.format(result.status))
# 请求ID。请求ID是本次请求的唯一标识，强烈建议在程序日志中添加此参数。
print('request_id: {0}'.format(result.request_id))
# ETag是put_object方法返回值特有的属性，用于标识一个Object的内容。
print('ETag: {0}'.format(result.etag))
# HTTP响应头部。
print('date: {0}'.format(result.headers['date']))          
```
上传Bytes


```
# 填写Object完整路径和Unicode字符。Object完整路径中不能包含Bucket名称。
bucket.put_object('exampleobject.txt', u'Hello OSS')   
```

上传网络流

OSS将网络流视为可迭代对象（Iterable），并以Chunked Encoding的方式上传。      
```
# requests.get返回的是一个可迭代对象（Iterable），此时Python SDK会通过Chunked Encoding方式上传。
# 填写网络流地址。
input = requests.get('http://www.aliyun.com')
# 填写Object完整路径。Object完整路径中不能包含Bucket名称。
bucket.put_object('exampleobject.txt', input)     
```

上传本地文件


### objectkey

objectkey, key以及objectname是同一概念，均表示对object执行相关操作时需要填写的object名称。例如向某一存储空间上传object时，objectKey表示上传的object所在存储空间的完整名称，即包含文件后缀在内的*完整路径*，如填写为abc/efg/123.jpg。

### Region(地域)

Region表示OSS的数据中心所在物理位置。         
eg:     
Region:华东1（杭州）           
Region ID:oss-cn-hangzhou          
Endpoint:oss-cn-hangzhou-internal.aliyuncs.com    
