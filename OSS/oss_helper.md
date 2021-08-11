# 基本概念

## 1. bucket(存储空间)

存储空间是用户用于object(存储对象)的容器，所有对象都隶属于某个存储空间。存储空间具有各种配置属性，包括地域、访问权限、存储类型等。

* 同一个存储空间的内部是扁平的，没有文件系统的目录等概念，所有的对象都直接隶属于其对应的存储空间。
* 每个用户可以拥有**多个存储空间**。同一阿里云账号在同一地域内创建的存储空间总数不能超过 100 个
* 存储空间的名称在OSS范围内必须是全局唯一的，一旦创建之后无法修改名称。
* 存储空间内部的对象数目没有限制。
* 存储空间一旦创建成功，名称和所处地域不能修改。

### 1.1 创建bucket

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

**objectkey**

objectkey, key以及objectname是同一概念，均表示对object执行相关操作时需要填写的object名称。例如向某一存储空间上传object时，objectKey表示上传的object所在存储空间的完整名称，即包含文件后缀在内的*完整路径*，如填写为abc/efg/123.jpg。

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

```
# 填写Object完整路径和本地文件的完整路径。Object完整路径中不能包含Bucket名称。
# 如果未指定本地路径，则默认从示例程序所属项目对应本地路径中上传文件。
bucket.put_object_from_file('exampleobject.txt', 'D:\\localpath\\examplefile.txt') 
```

### 2.2 追加上传

追加上传是指通过append_object方法在已上传的追加类型文件(appendable object)末尾直接追加内容.     

```
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'https://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

# 设置首次上传的追加位置（Position参数）为0。
# <yourObjectName>填写不包含Bucket名称在内的Object的完整路径，例如example/test.txt。
result = bucket.append_object('<yourObjectName>', 0, 'content of first append')
# 如果不是首次上传，可以通过bucket.head_object方法或上次追加返回值的next_position属性，获取追加位置。
bucket.append_object('<yourObjectName>', result.next_position, 'content of second append')
```

### 2.3 断点续传上传

通过断点续传上传的方式将文件上传到OSS前，您可以指定断点记录点。上传过程中，如果出现网络异常或程序崩溃导致文件上传失败时，将从断点记录处继续上传未上传完成的部分。

```
oss2.resumable_upload()
```
参数：    
bucket
key:上传oss的文件名称    
filename:待上传的本地文件名称      
store:指定保存断点信息的目录      
headers:HTTP头部   
multipart_threshold:文件长度大于该值时，则使用分片上传     
part_size:分片大小.      
progress_callback:上传进度回调函数      
num_threads:并发上传线程数        

前三个参数必须存在.

```
# 若使用store指定了目录，则断点信息将保存在指定目录中。若使用num_threads设置并发上传线程数，请将oss2.defaults.connection_pool_size设置为大于或等于并发上传线程数。默认并发上传线程数为1。
oss2.resumable_upload(bucket, '<yourObjectName>', '<yourLocalFile>',
    store=oss2.ResumableStore(root='/tmp'),
    multipart_threshold=100*1024,
    part_size=100*1024,
    num_threads=4)
```

### 2.4 进度条

```
import os, sys
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')
# 当无法确定待上传的数据长度时，total_bytes的值为None。
def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate), end='')
        sys.stdout.flush()
# progress_callback为可选参数，用于实现进度条功能。
bucket.put_object('<yourObjectName>', 'a'*1024*1024, progress_callback=percentage)
```

## 3. 下载文件

### 3.1 流式下载

如果要下载的文件太大，或者一次性下载耗时太长，您可以通过流式下载，一次处理部分内容，直到完成文件的下载。    

以下代码用于将exampleobject.txt文件的流式数据下载到本地/User/localpath路径下的examplefile.txt。

```
import shutil
import oss2

auth = oss2.Auth('yourAccessKeyId', 'yourAccessKeySecret')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

# 填写Object的完整路径。Object完整路径中不能包含Bucket名称。
object_stream = bucket.get_object('exampleobject.txt')
with open('/User/localpath/examplefile.txt', 'wb') as local_fileobj:
    shutil.copyfileobj(object_stream, local_fileobj)       
```

以下代码用于将exampleobject.txt文件流式拷贝到另一个文件exampleobjectnew.txt中。

```
# 填写Object的完整路径。Object完整路径中不能包含Bucket名称。
object_stream = bucket.get_object('exampleobject.txt')
# 填写另一个Object的完整路径。Object完整路径中不能包含Bucket名称。
bucket.put_object('exampleobjectnew.txt', object_stream)       
```

### 3.2 下载到本地文件

```
# 下载OSS文件到本地文件。如果指定的本地文件存在会覆盖，不存在则新建。
#  <yourLocalFile>由本地文件路径加文件名包括后缀组成，例如/users/local/myfile.txt。
#  <yourObjectName>表示下载的OSS文件的完整名称，即包含文件后缀在内的完整路径，例如abc/efg/123.jpg。
bucket.get_object_to_file('<yourObjectName>', '<yourLocalFile>')
```

### 3.3 断点续传下载

与断点续传上传相似，参数也相同。   
```
oss2.resumable_download()
```

```
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

# 请将oss2.defaults.connection_pool_size设成大于或等于线程数，并将part_size参数设成大于或等于oss2.defaults.multiget_part_size。
oss2.resumable_download(bucket, '<yourObjectName>', '<yourLocalFile>',
  store=oss2.ResumableDownloadStore(root='/tmp'),
  multiget_threshold=20*1024*1024,
  part_size=10*1024*1024,
  num_threads=3)		
```
### 3.4 进度条

```
import os, sys
import oss2

def percentage(consumed_bytes, total_bytes):
    if total_bytes:
        rate = int(100 * (float(consumed_bytes) / float(total_bytes)))
        print('\r{0}% '.format(rate), end='')

        sys.stdout.flush()

# progress_callback是可选参数，用于实现进度条功能。
bucket.get_object_to_file('<yourObjectName>', '<yourLocalFile>', progress_callback=percentage)
```

## 4. 管理文件

### 4.1 判断文件是否存在

以下代码用于判断examplebucket中的exampleobject.txt文件是否存在。  

```
import oss2

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', 'examplebucket')

# 填写Object的完整路径，Object完整路径中不能包含Bucket名称。
exist = bucket.object_exists('exampleobject.txt')
if exist:
    print('object exist')
else:
    print('object not exist')        
```

### 4.2 管理文件访问权限

```
# 设置文件的访问权限。
bucket.put_object_acl('<yourObjectName>', oss2.OBJECT_ACL_PUBLIC_READ)
```
```
# 获取指定文件的访问权限：
print(bucket.get_object_acl('<yourObjectName>').acl)
```

### 4.3 列举文件

* GetBucket(ListObjects)涉及参数说明:
    * prefix:本次查询结果的前缀
    * delimiter:对文件名称进行分组的字符
    * marker:此次列举文件的起点
* GetBucketV2(ListObjectsV2)涉及参数说明:
    * prefix:本次查询结果的前缀
    * delimiter:对文件名称进行分组的字符
    * start_after:此次列举文件的起点
    * fetch_owner:指定是否在返回中包含owner信息(true/false)

列举指定存储空间下的10个文件      

```
# oss2.ObjectIterator or oss2.ObjectIteraterV2 
for b in islice(oss2.ObjectIterator(bucket), 10):
    print(b.key)
```

列举指定存储空间下所有文件       

```
# oss2.ObjectIterator or oss2.ObjectIteraterV2 
for obj in oss2.ObjectIterator(bucket):
    print(obj.key)
```

假设存储空间中有4个文件： oss.jpg、fun/test.jpg、fun/movie/001.avi、fun/movie/007.avi，正斜线（/）作为文件夹的分隔符。    

```
# 列举fun文件夹下的所有文件，包括子目录下的文件。oss2.ObjectIterator or oss2.ObjectIteraterV2 
for obj in oss2.ObjectIterator(bucket, prefix='fun/'):
    print(obj.key)
```

列举指定起始位置后的所有文件

```
# oss2.ObjectIterator or oss2.ObjectIteraterV2 
# 列举指定字符串之后的所有文件。即使存储空间中存在marker的同名object，返回结果中也不会包含这个object。
for obj in oss2.ObjectIterator(bucket, marker="x2.txt"):
    print(obj.key)
```

列举指定目录下的文件和子目录

OSS没有文件夹的概念，所有元素都是以文件来存储。创建文件夹本质上来说是创建了一个大小为0并以正斜线（/）结尾的文件。这个文件可以被上传和下载，控制台会对以正斜线（/）结尾的文件以文件夹的方式展示。

通过delimiter和prefix两个参数可以模拟文件夹功能：

* 如果设置prefix为某个文件夹名称，则会列举以此prefix开头的文件，即该文件夹下所有的文件和子文件夹（目录）。     
* 如果再设置delimiter为正斜线（/），则只列举该文件夹下的文件和子文件夹（目录）名称，子文件夹下的文件和文件夹不显示.      

```
# 列举fun文件夹下的文件与子文件夹名称，不列举子文件夹下的文件。
for obj in oss2.ObjectIterator(bucket, prefix = 'fun/', delimiter = '/'):
    if obj.is_prefix(): 
        print('directory: ' + obj.key)
    else:              
        print('file: ' + obj.key)
```

```
# 列举fun文件夹下的文件与子文件夹名称，不列举子文件夹下的文件。如果不需要返回owenr信息可以不设置fetch_owner参数。
for obj in oss2.ObjectIteratorV2(bucket, prefix = 'fun/', delimiter = '/', fetch_owner=True):
    # 通过is_prefix方法判断obj是否为文件夹。
    if obj.is_prefix():  # 判断obj为文件夹。
        print('directory: ' + obj.key)
    else:                # 判断obj为文件。
        print('file: ' + obj.key)
        print('file owner display name: ' + obj.owner.display_name)
        print('file owner id: ' + obj.owner.id)
```

获取指定目录下的文件大小

```
import oss2

def CalculateFolderLength(bucket, folder):
    length = 0
    for obj in oss2.ObjectIterator(bucket, prefix=folder):
        length += obj.size
    return length
    
# oss2.ObjectIterator or oss2.ObjectIteraterV2 
for obj in oss2.ObjectIterator(bucket, delimiter='/'):
    if obj.is_prefix():  # 判断obj为文件夹。
        length = CalculateFolderLength(bucket, obj.key)
        print('directory: ' + obj.key + '  length:' + str(length) + "Byte.")
    else: # 判断obj为文件。
        print('file:' + obj.key + '  length:' + str(obj.size) + "Byte.")
```
### 4.4 删除文件

删除单个文件      

```
# 删除文件。<yourObjectName>表示删除OSS文件时需要指定包含文件后缀在内的完整路径，例如abc/efg/123.jpg。
# 如需删除文件夹，请将<yourObjectName>设置为对应的文件夹名称。如果文件夹非空，则需要将文件夹下的所有object删除后才能删除该文件夹。
bucket.delete_object('<yourObjectName>')
```

删除多个文件

```
# 批量删除3个文件。每次最多删除1000个文件。
result = bucket.batch_delete_objects(['<yourObjectName-a>', '<yourObjectName-b>', '<yourObjectName-c>'])
# 打印成功删除的文件名。
print('\n'.join(result.deleted_keys))
```

删除指定前缀的文件

```
prefix = "<yourKeyPrefix>"

# 删除指定前缀的文件。
for obj in oss2.ObjectIterator(bucket, prefix=prefix):
    bucket.delete_object(obj.key)
```

### 4.5 拷贝文件

对于小于1GB的文件，您可以使用简单拷贝。     

```
bucket.copy_object('<yourSourceBucketName>', '<yourSourceObjectName>', '<yourDestinationObjectName>')
```

对于大于1GB的文件，需要使用分片拷贝（UploadPartCopy）。分片拷贝分为三步：

* 通过bucket.init_multipart_upload初始化分片拷贝任务。
* 通过bucket.upload_part_copy进行分片拷贝。除最后一个分片外，其它分片都要大于100KB。
* 通过bucket.complete_multipart_upload提交分片拷贝任务。

```
import oss2
from oss2.models import PartInfo
from oss2 import determine_part_size

auth = oss2.Auth('<yourAccessKeyId>', '<yourAccessKeySecret>')
bucket = oss2.Bucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', '<yourBucketName>')

src_object = '<yourSourceObjectName>'
dst_object = '<yourDestinationObjectName>'

total_size = bucket.head_object(src_object).content_length
part_size = determine_part_size(total_size, prefered_size=100*1024)

# 初始化分片
upload_id = bucket.init_multipart_upload(dst_object).upload_id
parts = []

# 逐个分片拷贝
part_number = 1
offset = 0
while offset < total_size:
    num_to_upload = min(part_size, total_size - offset)
    byte_range = (offset, offset + num_to_load - 1)
    
    result = bucket.upload_part_copy(bucket.bucket_name, src_object, byte_range, dst_object, upload_id, part_number)
    parts.append(PartInfo(part_number, result.etag))
    
    offset += num_to_upload
    part_number += 1
    
# 完成分片拷贝
# 指定x-oss-forbid-overwrite为false时，表示允许覆盖同名Object。
# 指定x-oss-forbid-overwrite为true时，表示禁止覆盖同名Object，如果同名Object已存在，程序将报错。
# headers = dict()
# headers["x-oss-forbid-overwrite"] = "true"
# bucket.complete_multipart_upload(dst_object, upload_id, parts, headers=headers)    #禁止覆盖同名
bucket.complete_multipart_upload(dst_object, upload_id, parts)
```

### 4.6 禁止覆盖同名文件

```
# 指定x-oss-forbid-overwrite为false时，表示允许覆盖同名Object。
# 指定x-oss-forbid-overwrite为true时，表示禁止覆盖同名Object，如果同名Object已存在，程序将报错。
headers["x-oss-forbid-overwrite"] = "true"
result = bucket.put_object('<yourObjectName>', 'content of object', headers=headers)

# HTTP返回码。
print('http status: {0}'.format(result.status))
# 请求ID。请求ID是请求的唯一标识，强烈建议在程序日志中添加此参数。
print('request_id: {0}'.format(result.request_id))
# ETag是put_object方法返回值特有的属性。
print('ETag: {0}'.format(result.etag))
# HTTP响应头部。
print('date: {0}'.format(result.headers['date']))
```
### Region(地域)

Region表示OSS的数据中心所在物理位置。         
eg:     
Region:华东1（杭州）           
Region ID:oss-cn-hangzhou          
Endpoint:oss-cn-hangzhou-internal.aliyuncs.com    
