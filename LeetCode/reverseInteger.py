'''
Given a 32-bit signed integer, reverse digits of an integer.

Example 1:

Input: 123
Output: 321
Example 2:

Input: -123
Output: -321
Example 3:

Input: 120
Output: 21
Note:
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the reversed integer overflows.
'''

class Solution:
    def reverse(self, x: int) -> int:
        rev = int(str(abs(x))[::-1])
        if rev.bit_length() < 32:
            if x < 0:
                return -rev
            else:
                return rev
        else:
            return 0

        #return (-rev if x<0 else rev) if rev.bit_length < 32 else 0
