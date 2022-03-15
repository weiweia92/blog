class Solution:
    def twoSum(self, nums, target):
        hashtable = {}
        for i, num in enumerate(nums):
            if target - num not in hashtable:
                hashtable[num] = i 
            else:
                return hashtable[target-num], i
        
if __name__=='__main__':
    nums = [3,3]
    target = 6
    solution = Solution()
    output = solution.twoSum(nums, target)
    print(output)