'''
Example:

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
'''

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if len(digits) == 0:
            return []

        #digit_map = {'2':['a', 'b', 'c'], '3':['d', 'e', 'f'], '4':['g', 'h', 'i'],
        #             '5':['j', 'k', 'l'], '6':['m', 'n', 'o'], '7':['p', 'q', 'r', 's'],
        #             '8':['t', 'u', 'v'], '9':['w', 'x', 'y', 'z']}
        digit_map = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno',
                     '7':'pqrs','8':'tuv','9':'wxyz'}
        result = [""]
        for digit in digits:
            tmp_list = []
            for ch in digit_map[digit]:
                for str in result:
                    tmp_list.append(str + ch)
            result = tmp_list
        return result
        
