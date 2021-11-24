#输入描述：
#第一行输入数组元素个数
#第二行输入待排序的数组，每个数用空格隔开
#第三行输入一个整数0或1。0代表升序排序，1代表降序排序

#输出描述：
#输出排好序的数字
'''
输入：8
     1 2 4 9 3 55 64 25
     0
输出：1 2 3 4 9 25 55 64
'''

num = int(input())
L = map(int,input().split(' '))
flag = int(input())

if flag:
    result = map(str,sorted(L, reverse=True))
    print(' '.join(result))
else:
    result = map(str,sorted(L))
    print(' '.join(result))