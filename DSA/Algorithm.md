## 数据结构与算法
### 一.算法
#### 时间复杂度
常见时间复杂度:![](https://latex.codecogs.com/png.image?\dpi{110}%20O(1)%3CO(logn)%3CO(nlogn)%3Co(n^2)%3CO(n^2logn)%3CO(n^3))  
复杂问题的时间复杂度：![](https://latex.codecogs.com/png.image?\dpi{110}%20O(n!)%3CO(2^n)%3CO(n^n))  
快速判断算法复杂度哦（适用于绝大多数简单情况）：
1. 确定问题规模n
2. 循环减半过程->![](https://latex.codecogs.com/png.image?\dpi{110}%20logn)  
3. k层关于n的循环->![](https://latex.codecogs.com/png.image?\dpi{110}%20n^k)  
#### 空间复杂度
用来评估算法内存占用大小的式子  
空间复杂度的表示方法与时间复杂度完全一样  
算法使用几个变量:![](https://latex.codecogs.com/png.image?\dpi{110}%20O(1))  
算法使用了长度为n的一维列表:![](https://latex.codecogs.com/png.image?\dpi{110}%20O(n))  
算法使用了m行n列的二维列表：![](https://latex.codecogs.com/png.image?\dpi{110}%20O(mn))
“空间换时间”  
### 递归
1. 调用自身  
2. 结束条件  
![]()  

func1,func2不是递归：没有结束条件  
fun3,func4是递归，但是输出结果不同。当x=3时func3输出3，2，1；func4输出1，2，3.
#### 汉诺塔问题
n=2  
1. 把小圆盘从A移动到B
2. 把大圆盘从A移动到C
3. 把小圆盘从B移动到C  
![]()

n个盘子  
1. 把n-1个圆盘从A经过C移动到B
2. 把第n个圆盘从A移动到C
3. 把n-1个圆盘从B经过A移动到C  
![]()  
```
def hanoi(n, a, b, c):
    if n > 0:
        hanoi(n-1, a, c, b) #1
        print(f'moving from {a} to {c}') #2
        hanoi(n-1, b, a, c) #3
    
hanoi(3, 'A', 'B', 'C')
```
汉诺塔移动次数的递推式：![](https://latex.codecogs.com/png.image?\dpi{110}%20h(x)=2h(x-1)+1) .  #2^n  
### 查找
在一些数据元素中，通过一定的方法找出与给定关键字相同的数据元素的过程  
#### 列表查找(线性表查找)
从列表中查找指定元素    
input:列表，待查找元素  
output:元素下标（未找到元素时一般返回None或者-1）  
#### 内置列表查找元素:index()
index()用的是线性查找，不用二分查找的原因是二分查找有限制，它只针对有序列表，如果对于无序列表进行排序，排序的复杂度大于线性查找。
#### Linear Search(顺序查找)
```
# O(n)
def linear_search(li, val):
    for ind, v in enumerate(li):
        if v == val:
            return ind
    return None
```
#### Binary Search(二分查找)
从有序列表的初始候选去`li[0:n]`开始，通过对待查找的值与候选区中间值的比较，可以使候选区减少一半。  
```
#O(logn)
def binary_search(li, val):
    left = 0
    right = len(li) - 1
    while left <= right: #candidate regions have values
        mid = (left + right) // 2
        if li[mid] == value:
            return mid
        elif li[mid] > val: #待查找值在mid左侧
            right = mid - 1
        else:
            left = mid + 1
    else:
        return None
```
### 排序
一般计算机1s进行![](https://latex.codecogs.com/png.image?\dpi{110}%2010^7)次运算  
内置排序函数：`sort()`   
#### 1.Bubble Sort(冒泡排序)
1. 列表每两个相邻的数， 如果前面比后面大，则交换这两个数
2. 一趟排序完成后，则无序区减少一个数，有序区增加一个数 。总共用了n-1趟  

关键点：趟，无序区范围  
#如果某趟没有发生交换，则表明该序列已经排好序，可以中断循环
```
import random
#O(n^2)
def bubble_sort(li):
    for i in range(len(li)-1): #第i趟
        exchange = False
        for j in range(len(li)-i-1):
            if li[j] > li[j+1]:  #升序  降序将>改成<
                li[j], li[j+1] = li[j+1], li[j]
                exchange = True
		if not exchange:
        	return 
        
li = [random.randint(0, 10000) for i in range(10000)]
print(li)
bubble_sort(li) #n^2=10000*10000  time = n^2/10^7=10s
print(li)
```
#### 2.Select Sort(选择排序)
```
#O(n^2)
def select_sort_simple(li):
    li_new = []
    for i in range(len(li)):
        min_val = min(li)  #O(n)
        li_new.append(min_val)
        li.remove(min_val)  #O(n)
    return li_new
```
此方法不建议大家使用，原因为：生成了两个列表li_new, li，占内存。时间复杂度大  
```
#原地排序
def select_sort(li):
    for i in range(len(li)-1): #i是第i趟
        min_loc = i
        for j in range(i+1, len(li)):
            if li[j] < li[min_loc]:
                min_loc = j
        if min_loc != i:
        	li[i], li[min_loc] = li[min_loc], li[i]
```
1. 一趟排序记录最小的数，放到第一个位置
2. 再一趟排序记录列表无序区最小的数，放到第二个位置
...
算法关键点：有序区和无序区，无序区最小数的位置
3. Insert Sort(插入排序)  
![]()
```
#O(n^2)
def insert_sort(li):
    for i in range(1, len(li)): #i表示摸到的牌的下标
        tmp = li[i]
        j = i - 1 #j指的是手里的牌的下标
        #找插入的位置
        while j >= 0 and li[j] > tmp:  
            li[j+1] = li[j]
            j -= 1
        li[j+1] = tmp
```
这三种方法的效率不是很高，不太容易被大家接受，接下来是牛逼三人组
#### 4.快速排序
![]()
思路：
1. 取一个元素p（第一个元素），使元素p归位
2. 列表被p分成两部分，左边都比p小，右边都比p大
3. 递归完成排序  
```
import sys
sys.setrecursionlimit(100000) #修改递归的最大深度
def quick_sort(data, left, right):
    if left < right:
        mid = partition(data, left, right)
        quick_sort(data, left, mid-1)
        quick_sort(data, mid+1, right)
        
def partition(li, left, right): #O(n)
    tmp = li[left]
    while left < right:
        while left < right and li[right] >= tmp:  #从右边找比tmp小的值
            right -= 1 # 往左走一步
        li[left] = li[right]  #把右边的值写到左边空位上
        while left < right and li[left] <= tmp:
            left += 1
        li[right] = li[left]  #把左边的值写到右边空位上
    li[left] = tmp #把tmp归位
```
时间复杂度：![](https://latex.codecogs.com/png.image?\dpi{110}%20O(nlogn))  
最坏情况：->![](https://latex.codecogs.com/png.image?\dpi{110}%20O(n^2)) . (倒序的时候)  
#### 堆排序前传--树与二叉树
#### 树
1. 树是一种数据结构 例如：目录结构
2. 树是一种可以递归定义的数据结构
3. 树是由n个节点组成的集合：  
   * 如果n=0,则为一个空树
   * 如果n>0,则存在1个节点作为树的根节点，其他节点可以分为m个集合，每个集合本身又是一棵树  
![]() 

一些概念  
1. 根节点： A  叶子节点：B，C，H，I，P，Q，K，L，M，N
2. 树的深度：4
3. 树的度：eg:D节点的度为1，F节点的度为3
4. 孩子节点/父节点
5. 子树
```
#用树结构模拟文件系统
class Node:
    def __init__(self, name, type='dir'):
        self.name = name
        self.type = type # dir or file
        self.children = []
        self.parent = None
        
    def __repr__(self):
        return self.name
        
class FileSystemTree:
    def __init__(self):
        self.root = Node('/')
        self.now = self.root
        
    def mkdir(self, name):
        if name[-1] != '/':
            name += '/'
        node = Node(name)
        self.now.children.append(node)
        node.parent = self.node
        
    def ls(self):
        return self.now.children
    
    def cd(self, name):
        if name[-1] != '/':
            name += '/'
        if name =='../':
            self.now = self.now.parent
            return 
        for child in self.now.children:
            if child.name == name:
                self.now = child
                return
        raise ValueError('invalid dir')
```
#### 二叉树
1. 二叉树：度不超过2的树
2. 每个节点最多有两个孩子节点
3. 两个孩子节点被区分为左孩子节点和右孩子节点  

**满二叉树**：一个二叉树如果每一层的结点数都达到最大值，则这个二叉树就是满二叉树  
**完全二叉树**：叶节点只能出现在最下层和次下层，并且最下面一层的节点都集中在该层最左边的若干位置的二叉树  
![]()  
**二叉树的存储方式（表达方式）**  
链式存储方式  
**顺序存储方式-->列表存储**  
![]()  
#### 5.Heap Sort(堆排序)
堆：一种特殊的完全二叉树结构  
大根堆：一棵完全二叉树，满足任一节点都比其孩子节点大  
小根堆：一棵完全二叉树，满足任一节点都比其孩子节点小  
![]()  
**堆的向下调整**  
当根节点的左右子树都是堆时，可以通过一次向下调整来将其变换成一个堆。  
![]()  
堆排序过程  
1. 建立堆
2. 得到堆顶元素，为最大元素
3. 去掉堆顶，将堆最后一个元素放到堆顶，此时可通过一次调整重新使堆有序
4. 堆顶元素为第二大元素
5. 重复步骤3，直到堆为空  
```
def sift(li, low, high):
    '''
    li:list
    low:堆的根节点位置
    high:堆的最后一个节点的位置
    '''
    i = low #i最开始指向根节点
    j = 2 * i + 1 #j开始是左孩子
    tmp = li[low] #把堆顶存起来
    while j <= high:  #只要j位置有数
        if j + 1 <= high and li[j+1] > li[j]:  #如果右孩子有并且比较大
            j = j + 1 #j指向右孩子
        if li[j] > tmp:
            li[i] = li[j]
            i = j  #往下看一层
            j = 2 * i + 1
        else:  #tmp更大，把tmp放到i的位置上
            li[i] = tmp #把tmp放到某一级领导的位置
            break
   	else:
        li[i] = tmp  #把tmp放到叶子节点
        
def heap_sort(li):   
    n = len(li)
    for i in range((n-2)//2, -1, -1): #O(n/2)
        #i表示建堆的时候调整的部分的根的下标
        sift(li, i, n-1)  #O(logn)
    #建堆完成  （农村包围城市）
    for i in range(n-1, -1, -1):  #O(n)
        #i指向当前堆的最后一个元素
        li[0], li[i] = li[i], li[0]
        sift(li, 0, i-1) #i-1是新的high
    
```
堆排序的时间复杂度是nlogn  
**堆排序的内置模块heapq**  
```
import heapq #q->queue优先队列
import random
li = list(range(100))
random.shuffle(li)
print(li)
heapq.heapify(li) #建堆  默认小根堆
n = len(li)
for i in range(n):
    print(heapq.heappop(li), end=',')
```
**堆排序--topk问题**  
现有n个数，设计算法得到前k大的数（k<n）  
解决思路：  
1. 排序后切片 O(nlogn)
2. lowB三人组 O(kn) #走k趟
3. 堆排序思路 O(nlogk)  
解决思路:  
(1)取列表前k个元素建立一个小根堆。堆顶就是目前第k大的数  
(2)依次向后遍历原列表，对于列表中的元素，如果小于堆顶，则忽略该元素，如果大于堆顶，则将堆顶更换为该元素，并且对堆进行一次调整  
(3)遍历列表所有元素后，倒序弹出堆顶  
```
#比较排序
def sift(li,low,high): #小根堆
    '''
    li:list
    low:堆的根节点位置
    high:堆的最后一个元素的位置
    '''
    i = low #i最开始指向根节点
    j = 2 * i + 1 #j开始是左孩子
    tmp = li[low] #把堆顶存起来
    while j <= high:  #只要j位置有数
        if j + 1 <= high and li[j+1] < li[j]: 
            j = j + 1 #j指向右孩子
        if li[j] < tmp:
            li[i] = li[j]
            i = j #往下看一层
            j = 2 * i + 1
        else:  
            li[i] = tmp  
            break
    else:
        li[i] = tmp   #把tmp放到叶子节点上

def topk(li, k):
    #1.建堆
    heap = li[0:k]
    for i in range((k-2)//2, -1, -1):
        sift(heap, i, k-1)
    #2.遍历
    for i in range(k, len(li)-1):
        if li[i] > heap[0]:
            heap[0] = li[i]
            sift(heap, 0, k-1)
    #3.出数
    for i in range(k-1, -1, -1):
        heap[0], heap[i] = heap[i], heap[0]
        sift(heap, 0, i-1)
    return heap
```
#### 6.归并排序--使用归并
假设现在的列表分两段有序，如何将其合成一个有序列表，这种操作称为一次归并  
1. 分解：将列表越分越小，直至分成一个元素
2. 终止条件：一个元素是有序的
3. 合并：将两个有序列表归并，列表越来越大
#### NB三人组（快速排序，归并排序，堆排序）小结
1. 三种排序算法的时间复杂度都是O(nlogn)
2. 一般情况下，就运行时间而言：  
 快速排序<归并排序<堆排序
3. 三种排序算法的缺点：  
快速排序：极端情况下排序效率低（倒序）  
归并排序：需要额外的内存开销  
堆排序：在快的排序算法中相对较慢  
![]()  
#### 7.Shell Sort(希尔排序)
![]()  
1. 希尔排序是一种分组插p排序算法
2. 首先取一个整数![](https://latex.codecogs.com/png.image?\dpi{110}%20d_1=n/2)，将元素分为![](https://latex.codecogs.com/png.image?\dpi{110}%20d_1)个组，每组相邻两元素之间距离为![](https://latex.codecogs.com/png.image?\dpi{110}%20d_1)，在各组内进行直接插入排序
3. 取第二个整数![](https://latex.codecogs.com/png.image?\dpi{110}%20d_2=d_1/2),重复上述分p排序过程，直到![](https://latex.codecogs.com/png.image?\dpi{110}%20d_i=1)即z所有元素在同一组内进行直接插入排序
4. 希尔排序每趟并不使某些元素有序，而是使整个数据越来越接近有序，最后一趟排序使得所有数据有序  
```
def insert_sort_gap(li, gap):
    for i in range(gap, len(li)):  #i表示摸到的牌的下标
        tmp = li[i]
        j = i - gap #j指手里的牌的下标
        while j >= 0 and li[j] > tmp:
            li[j+gap] = li[j]
            j -= gap
        li[j+gap] = tmp

def shell_sort(li):
    d = len(li) // 2
    while d >= 1:
        insert_sort_gap(li, d)
        d //= 2
```
希尔排序的时间复杂度比较复杂，并且和选取的gap序列有关  
#### 8.Count Sort(计数排序)
对列表进行排序，已知列表中的数的范围都在0到100之间，设计时间复杂度为O(n)的算法  
```
#O(n)
def count_sort(li, max_count=100):
    count = [0 for _ in range(max_count+1)]
    for val in li:
        count[val] += 1
    li.clear()
    for ind, val in enumerate(count):
        for i in range(val):
            li.append(ind)
import random
li = [random.randint(0, 100) for _ in range(1000)]
print(li)
count_sort(li)
print(li)
```
计数排序比系统中的排序算法速度要快，但是使用起来有限制的，必须知道元素的范围，而且占用空间  
#### 9.Bucket Sort(桶排序)
1. 在计数排序中，如果元素的范b比较大（比如在1到1亿之间）如何改造算法？
2. 桶排序：首先将元素分在不同的桶中，在对每个每个桶中的元素进行排序  
![]()
```
def bucket_sort(li, n=100, max_num=10000):
    buckets = [[] for _ in range(n)] #创建桶
    for var in li:
        i = min(var//(max_num//n), n-1) #i表示var放到几号桶里
        buckets[i].append(var) #把var加到桶里
        #保持桶内的元素
        for j in range(len(buckets[i])-1, 0, -1):
            if buckets[i][j] < buckets[i][j-1]:
                buckets[i][j], buckets[i][j-1] = buckets[i][j-1], buckets[i][j]
            else:
                break
    sorted_li =[]
    for buc in buckets:
        sorted_li.extend(buc)
    return sorted_li
```
桶排序的表现取决于数据的分布。也就需要对不同数据排序时采取不同的分桶策略  
平均情况时间复杂度:![](https://latex.codecogs.com/png.image?\dpi{110}%20O(n+k))  
最坏情况时间复杂度：![](https://latex.codecogs.com/png.image?\dpi{110}%20O(n^2k))  
空间复杂度：![](https://latex.codecogs.com/png.image?\dpi{110}%20O(nk))    
