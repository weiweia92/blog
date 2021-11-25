#实现删除字符串中出现次数最少的字符，若多个字符出现次数一样，则都删除。输出删除这些单词后的字符串，字符串中其它字符保持原来的顺序。
#注意每个输入文件有多组输入，即多个字符串用回车隔开

characters = input()
strings = list(set(characters))
dct = {}
for i in strings:
    num = characters.count(i)
    dct[i] = num
sort_dct = sorted(dct.items(),key=lambda x:-x[1])

for item in sort_dct:
    if item[1] == sort_dct[-1][1]:
        characters = characters.replace(item[0],'')
print(characters)



