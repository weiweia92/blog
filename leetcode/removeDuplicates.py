class Solution:
    def removeDuplicates(self, nums):
        position = 1
        while position < len(nums):
            if nums[position] == nums[position - 1]:
                nums.pop(position)
            else:
                position += 1
        return len(nums)

    def removeElement(self, nums):
        # 给你一个数组 nums 和一个值 val，你需要原地移除所有数值等于 val 的元素，并返回移除后数组的新长度。
        position = 0
        while position < len(nums):
            if nums[position] ==  val:
                nums.pop(position)
            else:
                position += 1
        return len(nums)