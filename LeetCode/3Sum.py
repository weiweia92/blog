'''
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note:

The solution set must not contain duplicate triplets.

Example:

Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
'''

class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        result = []
        nums.sort()

        for i in range(n-2):
            #[0, 1, 2, 3, 4,...]
            #0+1+2=3>0
            if nums[i] + nums[i+1] + nums[i+2] > 0:
                break
            #[-100, 0, 1, 2, 3,...,8, 9]
            #-100+8+9=-83<0
            if nums[i] + nums[n-2] + nums[n-1] < 0:
                continue
            if 0 < i and nums[i] == nums[i-1]:
                continue #duplicate
            l, r = i+1, n-1
            while l < r:
                tmp = nums[i] + nums[l] + nums[r]
                if tmp == 0:
                    result.append([nums[i], nums[l], nums[r]])
                    while l+1 < r and nums[l] == nums[l+1]:
                        l += 1
                    l += 1
                    while r-1 > l and nums[r] == nums[r-1]:
                        r -= 1
                    r -= 1
                elif tmp < 0:
                    l += 1
                else:
                    r -= 1
                    
        return result

