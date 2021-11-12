'''
Example 1:

Input: "42"
Output: 42
Example 2:

Input: "   -42"
Output: -42
Explanation: The first non-whitespace character is '-', which is the minus sign.
             Then take as many numerical digits as possible, which gets 42.
Example 3:

Input: "4193 with words"
Output: 4193
Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.
Example 4:

Input: "words and 987"
Output: 0
Explanation: The first non-whitespace character is 'w', which is not a numerical 
             digit or a +/- sign. Therefore no valid conversion could be performed.
Example 5:

Input: "-91283472332"
Output: -2147483648
Explanation: The number "-91283472332" is out of the range of a 32-bit signed integer.
             Thefore INT_MIN (−231) is returned.
'''

'''
eg:
48 = 4*10 + 8
520 = (5*10 + 2)*10 + 0
4678 = (((4*10+6)*10+7)*10+8)
'''
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.strip()

        if not str:
            return 0
        negative = False
        out = 0

        if str[0] == '-':
            negative = True
        elif str[0] == '+':
            negative = False
        elif not str[0].isdigit():
            return 0
        else:
            out = ord(str[0]) - ord('0')

        for i in range(1, len(str)):
            if str[i].isdigit():
                out = out*10 + (ord(str[i]) - ord('0'))
                if not negative and out >= 2147483647:
                    return 2147483647
                if negative and out >= 2147483648:
                    return -2147483648
            else:
                break
        if not negative:
            return out
        else:
            return -out
        
