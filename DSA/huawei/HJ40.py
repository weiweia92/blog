#输入一行字符，分别统计出包含英文字母、空格、数字和其它字符的个数。
'''
输入：1qazxsw23 edcvfr45tgbn hy67uj m,ki89ol.\\/;p0-=\\][

输出：26
     3
     10
     12
'''
string = input()
a, b, c, d = 0, 0, 0, 0
for i in string:
    if i.isalpha():
        a += 1
    elif i == ' ':
        b += 1
    elif i.isdigit():
        c += 1
    else:
        d += 1
print(a)
print(b)
print(c)
print(d) 