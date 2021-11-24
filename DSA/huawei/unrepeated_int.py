#输入一个 int 型整数，按照从右向左的阅读顺序，返回一个不含重复数字的新的整数。
#保证输入的整数最后一位不是 0 。
'''
输入：9876673
输出：37689
'''
a = input()
num = a[::-1]

d = []
for i in num:
    if i not in d:
        d.append(i)
print(''.join(d))