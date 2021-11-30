#自守数是指一个数的平方的尾数等于该数自身的自然数。例如：25^2 = 625，76^2 = 5776，9376^2 = 87909376。
# 请求出n(包括n)以内的自守数的个数
'''
输入：5
     2000
输出：3
     8
说明：
对于样例一，有0，1，5，这三个自守数 
'''
def zishoushu(num):
    pow2 = num ** 2
    if len(str(pow2)) == 1:
        cut = str(str(pow2)[:])
    else:
        cut = str(str(pow2)[-len(str(num)):])
    if cut == str(num):
        return True
    else:
        return False

while True:
    try:
        n = int(input())
        count = 0
        for i in range(n+1):
            if zishoushu(i) == True:
                count += 1
        print(count)
    except:
        break
