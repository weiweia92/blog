### 39. 组合总和
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ，找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ，并以列表形式返回。你可以按 任意顺序 返回这些组合。

candidates 中的 同一个 数字可以 无限制重复被选取 。如果至少一个数字的被选数量不同，则两种组合是不同的。 

对于给定的输入，保证和为 target 的不同组合数少于 150 个。
>输入: candidates = [2,3,5], target = 8            
>输出: [[2,2,2,2],[2,3,3],[3,5]]
```


```
### 43. 字符串相乘
>输入: num1 = "2", num2 = "3"          
>输出: "6"
```
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if '0' in [num1, num2]:
            return '0'
        res = [0] * (len(num1) + len(num2))
        num1, num2 = num1[::-1], num2[::-1]
        for i1 in range(len(num1)):
            for i2 in range(len(num2)):
                digit = int(num1[i1]) * int(num2[i2])
                res[i1 + i2] += digit
                res[i1 + i2 + 1] += res[i1 + i2] // 10
                res[i1 + i2] = res[i1 + i2] % 10
        res, beg = res[::-1], 0
        while beg < len(res) and res[beg] == 0:
            beg += 1
        res = map(str, res[beg:])
        return ''.join(res)
```
### 46. 全排列
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。
>输入：nums = [1,2,3]          
>输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]       
```
class Solution:
    def permute(self, nums):
        if len(nums) == 1:
            return [nums]
        ans = []
        for i, num in enumerate(nums):
            n = nums[:i] + [i+1:]
            for y in self.permute(n):
                ans.append([num] + y)
        return ans
```
### 50. Pow(x, n)
```
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        elif n < 0:
            return self.myPow(1/x, -n)
        elif n % 2 == 0:
            tmp = self.myPow(x, n/2)
            return tmp * tmp
        else:
            return x * self.myPow(x, n-1)
```
### 53.最大子数组和
>输入：nums = [-2,1,-3,4,-1,2,1,-5,4]          
>输出：6          
>解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。      
```
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        maxSub = nums[0]
        curSum = 0
        for n in nums:
            if curSum < 0:
                curSum = 0
            curSum += n
            maxSub = max(curSum, maxSub)
        return maxSub
```
### 54. 螺旋矩阵
![](pic/spiralOrder.png)
```
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        row = len(matrix)
        if row == 0 or len(matrix[0]) == 0:
            return []
        col = len(matrix[0])
        res = matrix[0]
        if row > 1:
            for i in range(1, row):
                res.append(matrix[i][col-1])
            
            for j in range(col-2, -1, -1):
                res.append(matrix[row-1][j])
            if col > 1:
                for i in range(row-2, 0, -1):
                    res.append(matrix[i][0])
        M = []
        for k in range(1, row - 1):
            t = matrix[k][1:-1]
            M.append(t)

        return res + self.spiralOrder(M)
```
### 55. 跳跃游戏
![](pic/canJump.png)
```
class Solution:
    def canJump(self, nums):
        reach = 0
        for i, n in enumerate(nums):
            if i > reach:
                return False
            reach = max(reach, i + n)
        return True
```
### 56. 合并区间
>输入：intervals = [[1,3],[2,6],[8,10],[15,18]]          
>输出：[[1,6],[8,10],[15,18]]          
>解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].      
```

```

### 58. 最后一个单词的长度
>输入：s = "Hello World"             
>输出：5           
>解释：最后一个单词是“World”，长度为5。     
```
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        strings = s.strip().split(' ')
        return len(strings[-1])
```
### 66.加一
>示例 1：

>输入：digits = [1,2,3]        
>输出：[1,2,4]          
>解释：输入数组表示数字 123。         

>示例 2：

>输入：digits = [4,3,2,1]         
>输出：[4,3,2,2]           
>解释：输入数组表示数字 4321。        
```
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        str_digits = ''.join(list(map(str, digits)))
        process_digits = str(int(str_digits) + 1)
        ret = []
        for i in process_digits:
            ret.append(int(i))
        return ret
```
### 67. 二进制求和
>输入: a = "1010", b = "1011"       
>输出: "10101"
```
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a,2)+int(b,2))[2:]
```
### 69. x的平方根
给你一个非负整数 x ，计算并返回 x 的 算术平方根 。由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。
>输入：x = 8        
>输出：2           
>解释：8 的算术平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。
```
class Solution:
    def mySqrt(self, x):
        left = 0
        right = x
        while left <= right:
            mid = (left + right) // 2
            if mid * mid < x:
                left = mid + 1
            else:
                right = mid - 1
        return int(right)
```
### 70. 爬楼梯
Fibonacci sequence
```
class Solution:
    def climbStairs(self, n):
        prev, curr = 0, 1
        for i in range(n):
            prev, curr = curr, prev + curr
        return curr
```