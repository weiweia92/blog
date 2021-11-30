#矩阵乘法
#输入描述
#输入包含多组数据，每组数据包含：
#第一行包含一个正整数x，代表第一个矩阵的行数
#第二行包含一个正整数y，代表第一个矩阵的列数和第二个矩阵的行数
#第三行包含一个正整数z，代表第二个矩阵的列数
#之后x行，每行y个整数，代表第一个矩阵的值
#之后y行，每行z个整数，代表第二个矩阵的值

'''
输入：2
     3
     2
     1 2 3
     3 2 1
     1 2
     2 1
     3 3
输出：14 13
     10 11 
'''
x = int(input())
y = int(input())
z = int(input())
A = []
B = []
for i in range(x):
    A.append(list(map(int, input().split(' '))))
for j in range(y):
    B.append(list(map(int, input().split(' '))))
C = []
for i in range(x):
    for j in range(z):
        res = 0
        for k in range(y):
            res += A[i][k] * B[k][j]
        C.append(res)
for i in range(0, x):
    print(' '.join(map(str,C[i*z:(i+1)*z])))



