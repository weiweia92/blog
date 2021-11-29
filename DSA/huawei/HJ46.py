"""
输入描述：
1.输入待截取的字符串
2.输入一个正整数k，代表截取的长度

输出描述：
截取后的字符串
"""
strings, num = input(), int(input())
print(strings[:num])
