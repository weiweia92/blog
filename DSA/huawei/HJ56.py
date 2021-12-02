#完全数（Perfect number），又称完美数或完备数，是一些特殊的自然数。
#它所有的真因子（即除了自身以外的约数）的和（即因子函数），恰好等于它本身。
#例如：28，它有约数1、2、4、7、14、28，除去它本身28外，其余5个数相加，1+2+4+7+14=28。
'''
输入描述：输入一个数字n
输出描述：输出不超过n的完全数的个数
'''
def factorlist(num):
    if num == 1:
        return False
    L = []
    for i in range(1, num):
        if num % i == 0:
            L.append(i)
    return L

while True:
    try:
        n = int(input())
        count = 0
        for i in range(2,n+1):
            if sum(factorlist(i)) == i:
                count += 1
        print(count)
    except:
        break
