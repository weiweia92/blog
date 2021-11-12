'''
Example 1:

Input: "()"
Output: true
Example 2:

Input: "()[]{}"
Output: true
Example 3:

Input: "(]"
Output: false
Example 4:

Input: "([)]"
Output: false
Example 5:

Input: "{[]}"
Output: true
'''
class Solution:
    def isValid(self, s):
        stack = []
        mapping = {"(":")","{":"}","[":"]"}
        
        for char in s:
            if char in mapping:
                stack.append(char)
            elif len(stack) == 0 or mapping[stack.pop()] != char:
                return False
            # [{]}
            else:
                return False
        
        return len(stack) == 0 #'['
