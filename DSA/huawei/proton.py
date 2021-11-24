#功能:输入一个正整数，按照从小到大的顺序输出它的所有质因子（重复的也要列举）（如180的质因子为2 2 3 3 5 ）

'''
输入描述：
输入一个整数

输出描述：
按照从小到大的顺序输出它的所有质数的因子，以空格隔开。最后一个数后面也要有空格。
'''
import math

line = int(input())
i = 2
n = math.sqrt(line)
while i <= n and line > 1:
    if line % i == 0:
        print(i, end=' ')
        line /= i
        continue
    i += 1