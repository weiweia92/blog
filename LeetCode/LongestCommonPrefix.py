'''
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
Note:

All given inputs are in lowercase letters a-z.
'''

class Solution:
    def longestCommonPrefix(self, strs):
        if any(map(lambda x:len(x) == 0, strs)) or not strs:
            return ''
        if len(strs) == 1:
            return strs[0]

        minlen = min((word for word in strs), key=len)
        #minlen = len(str[0])
        #for i in range(len(strs)):
        #    minlen = min(len(strs[i]), minlen)

        lcp = ''
        i = 0
        while i < minlen:
            char = strs[0][i]
            for j in range(1, len(strs)):
                if strs[j][i] != char:
                    return lcp

            lcp = lcp + char
            i += 1

        return lcp
