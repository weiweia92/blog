'''
Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.

Example:

Input: [1,8,6,2,5,4,8,3,7]
Output: 49
'''
class Solution(object):
    #method1:bruce force
    #Time:O(n^2)
    #Space:O(1)
    def bruce_force(self, height):
        best = 0
        for l in range(len(height)):
            for r in range(l+1, len(height)):
                best = max(best, min(height[l], height[r]) * (r-l))
        return best
    
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height) - 1
        result = 0

        while left < right:
            water = min(height[left], height[right]) * (right - left)

            if water > result:
                result = water

            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return result
    
