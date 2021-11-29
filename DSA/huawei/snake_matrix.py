"""
蛇形矩阵是由1开始的自然数依次排列成的一个矩阵上三角形。
例如，当输入5时，应该输出的三角形为：
1 3 6 10 15
2 5 9 14
4 8 13
7 12
11
"""
num = int(input())
for i in range(num):
    if i == 0:
        res = [(i+2)*(i+1)//2 for i in range(num)]
    else:
        res = [x-1 for x in res[1:]]
    print(' '.join(map(str, res)))




