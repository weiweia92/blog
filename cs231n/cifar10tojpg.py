import numpy as np
import os

root_path_train = '/home/weiweia92/Downloads/cifar-10-python/cifar-10-batches-py/train/'
root_path_test = '/home/weiweia92/Downloads/cifar-10-python/cifar-10-batches-py/test/'

def Mkdir(path):
    i = 0
    for i in data_meta[b'label_names']:
        i = i.decode('utf-8')
        dir_name = path + str(i)
        os.mkdir(dir_name)

Mkdir(root_path_train)
Mkdir(root_path_test)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

filename = '/home/weiweia92/Downloads/cifar-10-python/cifar-10-batches-py'

meta = unpickle(filename+'/batches.meta')
label_name = meta[b'label_names']

for i in range(1,6):
    content = unpickle(filename+'/data_batch_'+str(i))
    print('load data...')
    print(content.keys())
    print('tranfering data_batch' + str(i))
    for j in range(10000):
        img = content[b'data'][j]
        img = img.reshape(3,32,32)
        img = img.transpose(1,2,0)
        img_name = '/home/weiweia92/Downloads/cifar-10-python/cifar-10-batches-py/train/'+label_name[content[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(j) +'.jpg'
        imageio.imwrite(img_name, img)

content_test = unpickle(filename+'/test_batch')
for j in range(10000):
    img = content_test[b'data'][j]
    img = img.reshape(3,32,32)
    img = img.transpose(1,2,0)
    img_name = '/home/weiweia92/Downloads/cifar-10-python/cifar-10-batches-py/test/'+label_name[content_test[b'labels'][j]].decode() + '/'+str(j) +'.jpg'
    imageio.imwrite(img_name, img)
