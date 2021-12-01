# 按照出入栈处理括号
while True:
    try:
        n = int(input())
        matrix_dict = {}
        for i in range(n):
            matrix_dict[chr(ord('A')+i)] = list(map(int, input().strip().split(' ')))
        s = input()
        result = 0
        temp = []
        for i in s:
            # 不遇到')'就一直压栈
            if i != ')':
                temp.append(i)
            else:
                C = temp.pop()
                B = temp.pop()
                temp.pop() # 弹出'('
                result += matrix_dict[B][0] * matrix_dict[B][1] * matrix_dict[C][1]
                