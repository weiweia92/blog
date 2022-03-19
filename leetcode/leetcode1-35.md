## leetcode 前100道题（easy + medium）

这里总结的主要是我理解的方法

### 1.两数之和
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值 target 的那两个整数，并返回它们的数组下标。你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。你可以按任意顺序返回答案。
>输入：nums = [2,7,11,15], target = 9        
>输出：[0,1]        
>解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 
```
class Solution:
    def twoSum(self, nums, target):
        hashtable = {}
        for i, num in enumerate(nums):
            if target - num not in hashtable:
                hashtable[num] = i
            return hashtable[target - num], i
```
### 2.两数相加

![](pic/listnode.png)

```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1, l2):
        curr = dummy = ListNode()
        count = 0
        while l1 or l2 or count:
            num = 0
            if l1:
                num += l1.val
                l1 = l1.next
            if l2:
                num += l2.val
                l2 = l2.next
            count, num = divmod(num + count, 10)
            curr.next = ListNode(num)
            curr = curr.next
        return dummy.next
```
### 3. 无重复字符的最长子串
给定一个字符串 s ，请你找出其中不含有重复字符的 最长子串 的长度。
>输入: s = "abcabcbb"        
>输出: 3          
>解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。           
```
class Solution:
    def lengthOfLongestSubstring(self, s):
        ans = ''
        tmp = ''
        for i in s:
            if i not in s:
                tmp += i
            else:
                tmp = tmp[tmp.index(i)+1:] + i
            if len(tmp) > len(ans):
                ans = tmp
        return len(ans)
```
### 4.寻找两个正序数组的中位数 
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的中位数 
>示例 1：        
>输入：nums1 = [1,3], nums2 = [2]        
>输出：2.00000     
>
>输入：nums1 = [1,2], nums2 = [3,4]         
>输出：2.50000
```
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        nums = nums1 + nums2
        nums.sort()
        if len(nums) % 2 == 0:
            index = int(len(nums) / 2)
            return (nums[index-1]+nums[index])/2.0
        else:
            index = int((len(nums)+1)/2)
            return nums[index-1]
```
### 5. 最长回文子串
给你一个字符串 s，找到 s 中最长的回文子串。
>输入：s = "babad"         
>输出："bab"
```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res = ''
        resLen = 0
        for i in range(len(s)):
            # odd length
            l, r = i, i
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    res = s[l: r+1]
                    resLen = r - l + 1
                l -= 1
                r += 1
            # even length
            l, r = i, i+1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if (r - l + 1) > resLen:
                    res = s[l: r+1]
                    resLen = r - l + 1
                l -= 1
                r += 1
        return res
```
### 6. Z字形变换
将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。
比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：
>P   A   H   N      
>A P L S I I G        
>Y   I   R            
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。
```
class Solution:
    def convert(self, s, numRows):
        if numRows == 1:
            return s
        row_map = {row:"" for row in range(1,numRows+1)}
        row = 1
        up = True
        for letter in s:
            row_map[row] += letter
            if (row == 1) or ((row < numRows and up)):
                row += 1
                up = True
            else:
                row -= 1
                up = False
        convert = ''
        for row in range(1, numRows+1):
            convert += row_map[row]
        return convert
```
### 7.整数反转
给你一个 32 位的有符号整数 x ，返回将 x 中的数字部分反转后的结果。
如果反转后整数超过 32 位的有符号整数的范围 $[−2^{31},  2^{31} − 1]$ ，就返回 0。
```
class Solution:
    def reverse(self, x: int) -> int:
        if x >= 0:
            x = int(str(x)[::-1])
        else:
            x = str(x)[1:][::-1]
            x = '-' + x
            x = int(x)
        if abs(x) > 2 ** 31:
            return 0
        else:
            return x
```
### 8.字符串转换整数(atoi)
请你来实现一个 myAtoi(string s) 函数，使其能将字符串转换成一个 32 位有符号整数（类似 C/C++ 中的 atoi 函数）。

函数 myAtoi(string s) 的算法如下：

1. 读入字符串并丢弃无用的前导空格
2. 检查下一个字符（假设还未到字符末尾）为正还是负号，读取该字符（如果有）。 确定最终结果是负数还是正数。 如果两者都不存在，则假定结果为正。
3. 读入下一个字符，直到到达下一个非数字字符或到达输入的结尾。字符串的其余部分将被忽略。
4. 将前面步骤读入的这些数字转换为整数（即，"123" -> 123， "0032" -> 32）。如果没有读入数字，则整数为 0 。必要时更改符号（从步骤 2 开始）。
5. 如果整数数超过 32 位有符号整数范围 $[−2^{31},  2^{31} − 1]$ ，需要截断这个整数，使其保持在这个范围内。具体来说，小于 $[−2^{31}$ 的整数应该被固定为 $[−2^{31}$ ，大于 $2^{31} − 1]$ 的整数应该被固定为 $2^{31} − 1]$ 。
返回整数作为最终结果。
注意：
本题中的空白字符只包括空格字符 ' ' 。
除前导空格或数字后的其余字符串外，请勿忽略 任何其他字符。
```
import re
class Solution:
    def myAtoi(self, s: str) -> int:
        return max(min(int(*re.findall('^[\+\-]?\d+', s.lstrip())), 2**31 - 1), -2**31)
```
### 9. 回文数
回文数是指正序（从左向右）和倒序（从右向左）读都是一样的整数。
```
class Solution:
    def isPalindrome(self, x: int) -> bool:
        y = str(x)[::-1]
        if str(x) == y:
            return True
        else:
            return False
```
### 10. 正则表达式匹配
### 11. 盛最多水的容器
![](pic/question_11.jpeg)
```
class Solution:
    def maxArea(self, height):
        largest = 0
        start = 0
        end = len(height) - 1
        while (start != end):
            next_area = min(height[start], height[end])*(end-start)
            if next_area > largest:
                largest = next_area
            if height[start] < height[end]:
                start += 1
            else:
                end -= 1
        return largest 
```
### 12. 整数转罗马数字
![](pic/intToRoman.png)
```
class Solution:
    def intToRoman(self, num):
        value_map = {1000: "M",900: "CM",500: "D",400: "CD",100: "C",90: "XC",
                     50: "L",40: "XL",10: "X",9: "IX",5: "V",4: "IV",1: "I",}
        res = ''
        for i in value_map:
            count = num // i
            if count:
                res += count * value_map[i]
                num %= i
        return res
```
### 13. 罗马数字转整数
```
class Solution:
    def romanToInt(self, s: str) -> int:
        dic = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        sum = 0
        for i in range(len(s)-1):
            if dic[s[i]] < dic[s[i+1]]:
                sum -= dic[s[i]]
            else:
                sum += dic[s[i]]
        return sum+dic[s[-1]]
```
### 14. 最长公共前缀
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。         
>示例 1：
>输入：strs = ["flower","flow","flight"]     
>输出："fl"         
>示例 2：       
>输入：strs = ["dog","racecar","car"]          
>输出：""         
>解释：输入不存在公共前缀。       
```
class Solution:
    def longestCommonPrefix(self, strs):
        commonPrefix = ''
        length = min(len(word) for word in strs)
        strs = [word.lower() for word in strs]
        for i in range(length):
            dic = set([word[i] for word in strs])
            if len(list(dic)) == 1:
                commonPrefix += strs[0][i]
            else:
                break
        return commonPrefix
```
### 15. 三数之和
给你一个包含 n 个整数的数组 nums，判断 nums 中是否存在三个元素 a，b，c ，使得 a + b + c = 0 ？请你找出所有和为 0 且不重复的三元组。
>输入：nums = [-1,0,1,2,-1,-4]           
>输出：[[-1,-1,2],[-1,0,1]]
```
class Solution:
    def threeSum(self, nums: List[int]) -> int:
        nums.sort()
        res = []
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l = i + 1
            r = len(nums) - 1
            while l < r:
                threeSum = nums[i] + nums[l] + nums[r]
                if threeSum == 0:
                    res.append([nums[i], nums[l], nums[r]])
                    l += 1
                    while nums[l] == nums[l - 1] and l < r:
                        l += 1
                if threeSum < 0:
                    l += 1
                else:
                    r -= 1
        return res
```
### 16. 最接近的三数之和
给你一个长度为 n 的整数数组 nums 和 一个目标值 target。请你从 nums 中选出三个整数，使它们的和与target 最接近。返回这三个数的和。假定每组输入只存在恰好一个解。
>输入：nums = [-1,2,1,-4], target = 1          
>输出：2           
>解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。             
```
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        res = sum(nums[:3])
        for i in range(len(nums) - 2):
            l = i + 1
            r = len(nums) - 1
            while l < r:
                threeSum = nums[i] + nums[l] + nums[r]
                if abs(threeSum - target) < abs(res - target):
                    res = threeSum
                if threeSum < target:
                    l += 1
                else:
                    r -= 1
        return res
```
### 17. 电话号码的字母组合
![](pic/letterCombination.png)
```
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        phone_map = {'2':'abc','3':'def','4':'ghi','5':'jkl','6':'mno','7':'pqrs','8':'tuv','9':'wxyz'}
        if len(digits) == 0:
            return []
        numbers = list(phone_map[digits[0]])
        for digit in digits[1:]:
            numbers = [old+new for old in numbers for new in list(phone_map[digit])]
        return numbers
```
### 18. 
### 19. 删除链表的倒数第N个节点
![](pic/removeNthFromEnd.png)
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def removeNthFromEnd(self, head, n):
        if not head:
            return head
        slow = ListNode()
        slow.next = head
        fast = slow
        for i in range(n):
            fast = fast.next
        while (fast.next != None):
            slow = slow.next
            fast = fast.next
        if slow.next = head:
            head = head.next
        else:
            slow.next = slow.next.next
        return head
```
### 20. 有效的括号
>输入：s = "()"      
>输出：true        
>输入：s = "([)]"          
>输出：false      
```
class Solution:
    def isValid(self, s):
        stack = []
        for i in s:
            if i in '([{':
                stack.append(i)
            if i == ')' and stack and stack[-1] == '(':
                stack.pop()
            if i == ']' and stack and stack[-1] == '[':
                stack.pop()
            if i == '}' and stack and stack[-1] == '{':
                stack.pop()
            else:
                return False
        if stack:
            return False
        else:
            return True
```
### 21.合并两个有序链表
![](pic/mergeTwoLists.png)
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, list1, list2):
        curr = dummy = ListNode(0)
        while list1 and list2:
            if list1.val > list2.val:
                curr.next = list2
                list2 = list2.next
            else:
                curr.next = list1
                list1 = list1.next
            curr = curr.next
        curr.next = list1 or list2
        return dummy.next
```
### 22. 括号生成
数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。
>输入：n = 3
>输出：["((()))","(()())","(())()","()(())","()()()"]
```
class Solution:
    def generateParenthesis(self, n):
        # only add open paranthesis if open < n
        # only add close paranthesis if closed < open
        # valide IIF open == closed == n
        stack = []
        res = []
        def backtrack(openN, closeN):
            if openN == closeN == n:
                res.append(''.join(stack))
                return
            if openN < n:
                stack.append('(')
                backtrack(openN + 1, closeN)
                stack.pop()
            if closeN < openN:
                stack.append(')')
                backtrack(openN, closeN + 1)
                stack.pop()
        backtrack(0, 0)
        return res
```
### 23. 合并K个升序链表
>输入：lists = [[1,4,5],[1,3,4],[2,6]]           
>输出：[1,1,2,3,4,4,5,6]           
>解释：链表数组如下：          
>[           
> 1->4->5,         
>  1->3->4,         
>  2->6         
>]          
>将它们合并到一个有序链表中得到。            
>1->1->2->3->4->4->5->6         
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if not lists:
            return
        if len(lists) == 1:
            return lists[0]
        mid = len(lists) // 2
        l = self.mergeKLists(lists[:mid])
        r = self.mergeKLists(lists[mid:])
        return self.merge(l, r)
    
    def merge(self, l1, l2):
        curr = dummy = ListNode(0)
        while l1 and l2:
            if l1.val <= l2.val:
                curr.nex t = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 or l2
        return dummy.next
```
### 24.两两交换链表中的节点
![](pic/swapPairs.png)
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        dummy = ListNode(0)
        res = dummy
        dummy.next = head
        while dummy.next and dummy.next.next:
            first = dummy.next
            second = dummy.next.next
            first.next = second.next
            second.next = first
            dummy.next = second
            dummy = dummy.next.next
        return res.next
```
### 25. K个一组翻转链表
给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
![](pic/reverseKGroup.png)
```
```
### 26. 删除有序数组中的重复项
>输入：nums = [0,0,1,1,1,2,2,3,3,4]        
>输出：5, nums = [0,1,2,3,4]
```
class Solution:
    def removeDuplicates(self, nums):
        position = 1
        while position < len(nums):
            if nums[position] == nums[position - 1]:
                nums.pop(position)
            else:
                position += 1
        return len(nums)
```
### 27. 移除元素
>输入：nums = [3,2,2,3], val = 3           
>输出：2, nums = [2,2]      
```
class Solution:
    def removeElement(self, nums, val):
        position = 0
        while position < len(nums):
            if nums[position] == val:
                nums.pop(position)
            else:
                position += 1
        return len(nums)
```
### 28. 实现strStr()
给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串出现的第一个位置（下标从 0 开始）。如果不存在，则返回  -1 。对于本题而言，当 needle 是空字符串时我们应当返回 0 。
```
class Solution:
    def strStr(self, haystack, needle):
        return 0 if needle == '' else haystack.find(needle)
```
### 29. 两数相除
给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。返回被除数 dividend 除以除数 divisor 得到的商。
```
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        #2147483647
        d = abs(dividend)
        dv = abs(divisor)
        output = 0
        while d >= dv:
            tmp = dv
            mul = 1
            while d >= tmp:
                d -= tmp
                output += mul
                mul += mul
                tmp += tmp
        if (dividend < 0 and divisor >= 0) or (divisor < 0 and dividend >= 0):
            output = -output
        return min(2**31-1,output)
```
### 31. 下一个排列
整数数组的 下一个排列 是指其整数的下一个字典序更大的排列。更正式地，如果数组的所有排列根据其字典顺序从小到大排列在一个容器中，那么数组的 下一个排列 就是在这个有序容器中排在它后面的那个排列。如果不存在下一个更大的排列，那么这个数组必须重排为字典序最小的排列（即，其元素按升序排列）。
>nums = [5,4,3,2,1]      
>out -> [1,2,3,4,5] 

>nums = [1,2,7,9,6,4,1]             
>out -> [1,2,9,1,4,6,7]        

>nums = [1,7,9,9,8,3]       
>out -> [1,8,3,7,9,9]
```
class Solution:
    def nextPermutation(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        N = len(nums)
        pivot = 0
        # find pivot
        for i in range(N-1, 0, -1):
            if nums[i-1] < nums[i]:
                pivot = i
                break
        if pivot == 0:
            nums.sort()
            return
        # then find the swap which first number > swap
        swap = N - 1
        while nums[pivot - 1] >= nums[swap]:
            swap -= 1
        # swap
        nums[pivot - 1], nums[swap] = nums[swap], nums[pivot - 1]
        nums[pivot:] = sorted(nums[pivot:])
```
### 32. 最长有效括号
```

```
给你一个只包含 '(' 和 ')' 的字符串，找出最长有效（格式正确且连续）括号子串的长度。
>输入：s = "(()"
>输出：2
>解释：最长有效括号子串是 "()"
### 33. 搜索旋转排序数组
整数数组 nums 按升序排列，数组中的值 互不相同 。

在传递给函数之前，nums 在预先未知的某个下标 k（0 <= k < nums.length）上进行了 旋转，使数组变为 [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]（下标 从 0 开始 计数）。例如， [0,1,2,4,5,6,7] 在下标 3 处经旋转后可能变为 [4,5,6,7,0,1,2] 。

给你 旋转后 的数组 nums 和一个整数 target ，如果 nums 中存在这个目标值 target ，则返回它的下标，否则返回 -1 。
>输入：nums = [4,5,6,7,0,1,2], target = 0            
>输出：4
```
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if target >= nums[0]:
                if nums[mid] >= nums[0] and target > nums[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if nums[mid] >= nums[0] or target > nums[mid]:
                    left += 1
                else:
                    right = mid - 1
        return -1
```
### 34.在排序数组中查找元素的第一个和最后一个位置
给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。如果数组中不存在目标值 target，返回 [-1, -1]。进阶：你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
>输入：nums = [5,7,7,8,8,10], target = 8             
>输出：[3,4]
```
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        left = self.binSearch(nums, target, True)
        right = self.binSearch(nums, target, False)
        return [left, right]

    def binSearch(self, nums, target, leftBias):
        l, r = 0, len(nums) - 1
        i = -1
        while l <= r:
            m = (l + r) // 2
            if target > nums[m]:
                l = m + 1
            elif target < nums[m]:
                r = m - 1
            else:
                i = m
                if leftBias:
                    r = m - 1
                else:
                    l = m + 1
        return i
```
### 35.搜索插入位置
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
>输入: nums = [1,3,5,6], target = 5          
>输出: 2       
```
class Solution:
    def searchInsert(self, nums, target):
        if target not in nums:
            nums.append(target)
        return sorted(nums).index(target)
```