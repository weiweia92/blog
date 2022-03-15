class Solution:
    def maxArea(self, height):
        start = 0
        end = len(height) - 1
        largest = 0
        while start != end:
            next_area = min(height[start], height[end]) * (end - start)
            if next_area > largest:
                largest = next_area
            if height[start] < height[end]:
                start += 1
            else:
                end -= 1
        return largest