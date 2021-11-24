'''
编写一个函数，计算字符串中含有的不同字符的个数。字符在 ASCII 码范围内( 0~127 ，包括 0 和 127 )，
换行表示结束符，不算在字符里。不在范围内的不作统计。多个相同的字符只计算一次

例如，对于字符串 abaca 而言，有 a、b、c 三种不同的字符，因此输出 3 。
'''

string = input()
L = []
for i in string:
    L.append(i)

l = list(set(L))
print(len(l))

'''
输入描述：
一个只包含小写英文字母和数字的字符串。

输出描述：
一个字符串，为不同字母出现次数的降序表示。若出现次数相同，则按ASCII码的升序输出。

输入：aaddccdc
     1b1bbbbbbbbb

输出：cda
     b1
'''
string = input()
dct = {}
keys = list(set(string))
for key in keys:
    dct[key] = string.count(key)
L = sorted(dct.items(), key=lambda x:(-x[1],x[0]))
res = []
for i in L:
    res.append(i[0])
print(''.join(res))
