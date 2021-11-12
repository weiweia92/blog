#string convert(string s, int numRows);

#Example 1:
#Input: s = "PAYPALISHIRING", numRows = 3
#Output: "PAHNAPLSIIGYIR"
#Explaination:
#P   A   H   N
#A P L S I I G
#Y   I   R

#Example 2:
#Input: s = "PAYPALISHIRING", numRows = 4
#Output: "PINALSIGYAHRPI"
#Explanation:

#P     I    N
#A   L S  I G
#Y A   H R
#P     I

class Solution:
    def convert(self, s: str, numRows: int) -> str:

        length = len(s)
        if numRows == 1 or numRows >= length:
            return s

        interval = 2*numRows - 2
        ret = ''
        for i in range(numRows):
            j = i
            step = interval - 2*i
            while j < length:
                ret += s[j]
                if i ==0 or i == numRows-1:
                    j += interval
                else:
                    j += step
                    step = interval - step

        return res
