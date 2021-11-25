#Lily使用的图片包括"A"到"Z"、"a"到"z"、"0"到"9"。输入字母或数字个数不超过1024。
#输出描述：
#Lily的所有图片按照从小到大的顺序输出（ASCII码值）

'''
sorted函数就是根据ASCII码值从小到大进行排序的
'''
print(''.join(sorted(input())))
