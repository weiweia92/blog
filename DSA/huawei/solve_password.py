#假设渊子原来一个BBS上的密码为zvbo9441987,为了方便记忆，他通过一种算法把这个密码变换成YUANzhi1987
#他是这么变换的，大家都知道手机上的字母： 1--1， abc--2, def--3, ghi--4, jkl--5, mno--6, pqrs--7, tuv--8 wxyz--9, 0--0,
#把密码中出现的小写字母都变成对应的数字，数字和其他的符号都不做变换
#密码中没有空格，而密码中出现的大写字母则变成小写之后往后移一位，如：X ，先变成小写，再往后移一位，是 y ，Z 往后移是 a .
'''
输入：YUANzhi1987
输出：zvbo9441987
'''
import ipdb
password = input()
dct = {'1':1,'abc':2,'def':3,'ghi':4,'jkl':5,'mno':6,'pqrs':7,'tuv':8,'wxyz':9,'0':0}

output = []
for i in password:
    if i.isupper() and i != 'Z':
        i = chr(ord(i.lower()) + 1)
    elif i == 'Z':
        i = 'a'
    elif i.islower():
        for key in dct.keys():
            if i in key:
                i = str(dct[key])
    else:
        pass
    output.append(i)
print(''.join(output))

        




