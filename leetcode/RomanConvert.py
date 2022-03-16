class Solution:
    def intToRoman(self, num):
        VALUE_SYMBOLS = {1000: "M",900: "CM",500: "D",400: "CD",100: "C",90: "XC",
                         50: "L",40: "XL",10: "X",9: "IX",5: "V",4: "IV",1: "I",}
        res = ""
        for i in VALUE_SYMBOLS:
            count = num // i
            if count:
                res + count * VALUE_SYMBOLS[i]
                num %= i
        return res
    
class Solution:
    def romanToInt(self, s):
        dic = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        sum = 0
        for i in range(len(s)-1):
            if dic[s[i]] < dic[s[i+1]]:
                sum -= dic[s[i]]
            else:
                sum += dic[s[i]]
        return sum+dic[s[-1]]