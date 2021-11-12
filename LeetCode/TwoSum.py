'''
Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

Explain:

d = {}
index, value  target-value  d
0      2      9-2=7         d[2]=0  {2:0}
1      7      9-7=2         return 0,1
2      11     9-11=-2       d[11]=2
3      15     9-15=-4       d[15]=3  {2:0,11:2,15:3}

'''

class Solution(object):
    #brute force way
    def bruteForceTwoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if (nums[i] + nums[j] == target):
                    return [i, j]

    def twoSum(self, nums, target):
        #hash method
        d = {}
        for index,value in enumerate(nums):
            if target-value in d:
                return d[target-value],index
            else:
                d[value] = index
                   

nums = [2, 7, 11, 15]
target = 9
                
solution = Solution()

index1,index2 = solution.twoSum(nums, target)
