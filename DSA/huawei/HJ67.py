# 24点游戏算法
# 给出4个1-10的数字，通过加减乘除运算，得到数字为24就算胜利,除法指实数除法运算,本题对数字选取顺序无要求，
# 但每个数字仅允许使用一次，且不考虑括号运算
'''
输入：7 2 1 10
输出：true
'''
import itertools as it
while True:
    try:
        a,b,c,d = [int(x) for x in input().split(" ")]
        flag = False
        ls = [a,b,c,d]
        temp = list(it.permutations(ls)) #全排列，得到元组构成的列表
        #print(list(it.permutations(ls)))
        for i in temp:
            a,b,c,d = i[0],i[1],i[2],i[3]
            #print(a,b,c,d)
            first = [] #a和b运算
            second = [] #利用上面运算的结果再一次运算
            third = []  #利用上面运算的结果再一次运算
            first.append(a+b)
            first.append(a-b)
            first.append(a*b)
            first.append(a/b)
 
            #print(first)
 
            for i in first:
                second.append(i+c)
                second.append(i-c)
                second.append(i*c)
                second.append(i/c)
            #print(second)
 
            for k in second:
                third.append(float(k+d))
                third.append(float(k-d))
                third.append(float(k*d))
                third.append(float(k/d))
            #print(third)
 
            if float(24) in third:#要用浮点双精度，否则可能报错
                flag = True
        if flag == True:
            print("true")
        else:
            print('false')
    except:
        break