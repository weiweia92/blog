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
    
class Solution(object):
    def RomanToInt(self, num):
        