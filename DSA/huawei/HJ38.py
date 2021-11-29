"""
假设一个球从任意高度自由落下，每次落地后反跳回原高度的一半; 再落下, 求它在第5次落地时，共经历多少米?第5次反弹多高？
最后的误差判断是小数点6位
"""
h = int(input())
distance = h
for i in range(1, 5):
    distance += 2* h/2
    h = h / 2
print(distance)
print(h/2)
