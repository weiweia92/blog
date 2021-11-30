#任意一个偶数（大于2）都可以由2个素数组成，组成偶数的2个素数有很多种情况，本题目要求输出组成指定偶数的两个素数差值最小的素数对。
'''
输入：20
输出：7
     13
'''
import math

def isPrime(n):
    for i in range(2, int(math.sqrt(n))+1):
        if n % i == 0:
            return False
    return True

# 从中间往两边扩
while True:
    try:
        num, start = int(input()) // 2, 1
        if num % 2 == 1:
            start = 0
        for i in range(start, num, 2):
            a, b = num + i, num - i
            if isPrime(a) and isPrime(b):
                print(b)
                print(a)
                break
    except:
        break