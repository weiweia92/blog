import ipdb

def str2ASCII(string):
    res = [0] * 6
    if len(string) == 0:
        return []
    else:
        string = list(string)
        count, rem = divmod(len(string), 6)
        matrix = [[] * 6 for i in range(count)]
        for i in range(count):
            for j in range(6):
                matrix[i][j] = string[i * 6 + j]
        for i in range(len(res)):
            res[i] = [sum(ord(matrix[c][i])) for c in range(count)]
    return res

def multi2one(num):
    num = list(str(num))
    while len(num) > 1:
        num = str(sum(map(int, num)))
    return num

def scale_number(nums):
    final = []
    for num in nums:
        final.append(multi2one(num))
    return ''.join(final)

if __name__=='__main__':
    string = 'zhangfeng'
    tmp = str2ASCII(string)
    ret = scale_number(tmp)
    print(ret)

