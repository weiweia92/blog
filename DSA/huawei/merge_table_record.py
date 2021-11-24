#数据表记录包含表索引和数值（int范围的正整数），请对表索引相同的记录进行合并，即将相同索引的数值进行求和运算，
#输出按照key值升序进行输出。
'''
输入：4
     0 1
     0 2
     1 2
     3 4

输出：0 3
     1 2
     3 4
'''

n = {}
num = int(input().strip())
for _ in range(num):
    index, value = map(int,input().split(' '))
    if index in n.keys():
        n[index] = n[index] + value
    else:
        n[index] = value
for i in sorted(n.keys()):
    print(f'{i} {n[i]}')