#Example 1:
#Input: "abcabcbb"
#Output: 3 
#Explanation: The answer is "abc", with the length of 3. 

#Example 2:
#Input: "bbbbb"
#Output: 1
#Explanation: The answer is "b", with the length of 1.

#Example 3:
#Input: "pwwkew"
#Output: 3
#Explanation: The answer is "wke", with the length of 3. 
#             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """
        :type s:str
        :rtype: int
        """
        dic = {}
        left = -1 # point to the leftmost previous of the slide window
        res = 0
        for i in range(len(s)):
            if s[i] in dic and dic[s[i]] > left:
                left = dic[s[i]]
            dic[s[i]] = i
            res = max(res, i-left)
        return res

#explain:
#'pwwkew'
#init    i=0          i=1            i=2                   i=3                     i=4
#dic={}  {'p':0}      {'p':0,'w':1}  {'p':0,'w':1,'w':2}  {'p':0,'w':1,'w':2,'k':3} {'p':0,...'e':4}
#left=-1 -1           -1              1                     1                       1
#res=0   max(0,0-(-1)) max(1,1-(-1))  max(2,2-(1))         max(1,1-1)                max(1,4-1)
#        =1            =2             =1                   =2                        =3
#i=5
#{'p':0,...,'w':5}
#2
#max(3,4-2)
#=3
