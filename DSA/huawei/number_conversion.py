# 输入一个十六进制的数值字符串。注意：一个用例会同时有多组输入数据
'''
输入：0xA
     0xAA

输出：10
     170
'''
import sys

for line in sys.stdin:
    output = int(line.strip(),16)
    print(output)