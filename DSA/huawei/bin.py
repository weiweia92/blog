'''
输入描述：
 输入一个整数（int类型）

输出描述：
 这个数转换成2进制后，输出1的个数
'''
num = bin(int(input()))
string = str(num)
number = string.count('1')
print(number)