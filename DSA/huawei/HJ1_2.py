# 计算字符串最后一个单词的长度，单词以空格隔开，字符串长度小于5000。
'''
输入：hello nowcoder
输出：8
'''
words = input().split(' ')
print(len(words[-1]))

#输入描述：
#第一行输入一个由字母和数字以及空格组成的字符串，第二行输入一个字符。

#输出描述：
#输出输入字符串中含有该字符的个数。（不区分大小写字母）
'''
输入：ABCabc
     A
输出：2
'''
while True:
    try:
        strings = input().lower()
        string = input().lower()
        print(strings.count(string))
    except:
        break