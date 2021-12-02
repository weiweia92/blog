#定义一个单词的“兄弟单词”为：交换该单词字母顺序（注：可以交换任意次），而不添加、删除、修改原有的字母就能生成的单词。
#兄弟单词要求和原来的单词不同。例如： ab 和 ba 是兄弟单词。 ab 和 ab 则不是兄弟单词。
#现在给定你 n 个单词，另外再给你一个单词 str ，让你寻找 str 的兄弟单词里，按字典序排列后的第 k 个单词是什么？
'''
输入：3 abc bca cab abc 1
输出：2
     bca
'''
while True:
    try:
        line = input().split(' ')
        words = line[1:-2]
        target = line[-2]
        index = int(line[-1])
        bros = []
        for word in words:
            if word != target and sorted(word) == sorted(target):
                bros.append(word)
        print(len(bros))
        if len(bros) >= index:
            print(sorted(bros)[index - 1])
    except:
        break
