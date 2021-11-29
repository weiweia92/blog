#给出一个名字，该名字有26个字符组成，定义这个字符串的“漂亮度”是其所有字母“漂亮度”的总和。
#每个字母都有一个“漂亮度”，范围在1到26之间。没有任何两个不同字母拥有相同的“漂亮度”。字母忽略大小写。
#给出多个名字，计算每个名字最大可能的“漂亮度”。
"""
输入：2
     zhangsan
     lisi
输出：
     192
     101
"""
def beauty_score(name):
    set1 = set(name)
    dic = {}
    for i in set1:
        dic[i] = name.count(i)
    sorted_dic = sorted(dic.items(),key=lambda x:x[1], reverse=True)
    score = 0
    for j in range(len(sorted_dic)):
        score += (26-j)*sorted_dic[j][1]
    return score

while True:
    try:
        num = int(input())
        names = []
        for i in range(num):
            names.append(input())
        for name in names:
            score = beauty_score(name)
            print(score)
    except:
        break

