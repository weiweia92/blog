class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        total_nums = nums1+nums2
        total_nums.sort()
        len_total = len(total_nums)
        if len_total%2!=0:
            return total_nums[len_total//2]
        else:
            midpoint = len_total//2
            return 0.5*(total_nums[midpoint-1]+total_nums[midpoint])
