class Solution:
    def strStr(self, haystack, needle):
        #给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1
        #对于本题而言，当 needle 是空字符串时我们应当返回 0 
        return 0 if needle == '' else haystack.find(needle)
 
class Solution:   
    def searchInsert(self, nums, target):
        # 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
        if target not in nums:
            nums.append(target)
        return sorted(nums).index(target)
