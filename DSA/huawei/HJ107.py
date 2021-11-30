#计算一个浮点数的立方根，不使用库函数。保留一位小数。pow只能对大于等于0情况
#数据范围：|val|<=20
def cube(num):
    sig = 1
    if num < 0:
        sig = -1
    num = abs(num)
    a = round(pow(num, 1/3), 1)
    return a*sig

while True:
    try:
        num = float(input())
        res = cube(num)
        print(res)
    except:
        break