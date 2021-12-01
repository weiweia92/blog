#要求：
#实现如下2个通配符：
#*：匹配0个或以上的字符（注：能被*和?匹配的字符仅由英文字母和数字0到9组成，下同）
#？：匹配1个字符

#注意：匹配时不区分大小写。
# buhui
import re
while True:
    try:
        s1 = input().lower()
        s2 = input().lower()
        s1 = s1.replace('?','\w{1}').replace('.','\.').replace('*','\w*')
