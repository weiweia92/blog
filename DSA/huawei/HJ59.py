#输入描述：
#输入几个非空字符串
#输出描述：
#输出第一个只出现一次的字符，如果不存在输出-1
'''
输入：asdfasdfo
     aabb
输出：o
     -1
'''

while True:
    try:
        strings = input()
        set1 = set(strings)
        dic = {}
        for i in set1:
            value = strings.count(i)
            dic[i] = value
        if 1 not in dic.values():
            print(-1)
        else:
            for j in strings:
                if dic[j] == 1:
                    print(j)
                    break
    except:
        break