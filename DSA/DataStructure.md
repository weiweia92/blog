## 数据结构
### 1.数组和列表
![]()  
#### 列表
插入和删除操作都是O(n)复杂度：删除一个元素后，后面的元素都会向前移  
查找和添加是O(1)复杂度  

数据结构按照其逻辑结构分为线性结构，树结构，图结构  
线性结构：数据结构中的元素存在一对一的相互关系  
树结构：数据结构中的元素存在一对多的相互关系  
图结构：数据结构中的元素存在多对多的相互关系  
### 2.Stack(栈)
栈是一个数据集合，可以理解为只能在一端进行插入或删除操作的列表  
* 栈的特点：last-in, first-out
* 栈的基本操作：进栈（压栈）push --li.append .  出栈:pop--li.pop, 取栈顶:gettop--li[-1]
* 使用一般的列表结构即可以实现栈  
![]()  
```
class Stack:
    
    def __init__(self):
        self.stack = []
        
    def push(self, element):
        self.stack.append(element)
        
    def pop(self):
        return self.stack.pop()
    
    def get_top(self):
        if len(self.stack) > 0:
            return self.stack[-1]
       	else:
            return None
    def is_empty:
        return len(self.stack) == 0
```
#### 栈的应用--括号匹配问题
eg:  
()()[]{}：匹配  
([{()}])：匹配  
[](：不匹配  
[(])：不匹配  
```
#沿用上面的Stack类
def brace_match(s):
    match = {'}':'{', ']':'[', ')':'('}
    stack = Stack()
    for char in s:
        if char in ['(', '[', '{']:
            stack.push(char)
        else:
            if stack.is_empty():
                return False
            elif stack.get_top() == match[char]:
                stack.pop()
            else: #stack.get_top() != match[char]
                return False
   	if stack.is_empty():
        return True
    else:
        return False
```
### 3.队列
* 队列（Queue）是一个数据集合，仅允许在列表的一端进行插入，另一端进行删除  
* 进行插入的一端称为队尾(rear)，插入动作称为进队或者入队 O(1)
* 进行删除的一端称为队头(front)，删除动作称为出队 . O(1)
* 队列性质：First-in,First-out  
![]()  
**队列能否用列表简单实现？为什么？**  
![]()  
入队时用列表的append就可以，但在出队时使用列表的remove,pop在最后面进行删除操作时，复杂度是O(1),但要是删除的是最前面的元素，后面的元素都会向前挪，导致复杂度增加变成了O(n)；图d情况下，入队的话，应用列表需要继续开辟空间，进行入队，但前面的空间就白白浪费了
#### 队列的实现方式--环形队列
![]()  
* 环形队列：当队尾指针front==maxsize-1时，再前进一个位置就自动到0
* 队首指针前进1:front=(front+1)%maxsize
* 队尾指针前进1:rear=(rear+1)%maxsize
* 队空条件：rear==front
* 队满条件：(rear+1)%maxsize==front  
```
 class Queue:
    def __init__(self, size=100):
        self.queue = [0 for _ in range(size)]
        self.size = size
        self.rear = 0 #队尾指针
        self.front = 0 #队头指针
        
    def push(self, element):
        if not self.is_filled():
        	self.rear = (self.rear + 1) % self.size
        	self.queue[self.rear] = element
        else:
    		raise IndexError('Queue is filled.')
    def pop(self):
        if not self.is_empty():
            self.front = (self.front + 1) % self.size
            return self.queue[self.front]
        else:
            raise IndexError('Queue is empty.')
            
    def is_empty(self):
        return self.rear == self.front
    
    def is_filled(self):
        return (self.rear + 1) % self.size == self.front
            
```
#### 队列的内置模块
**双向队列**   
双向队列的两端都支持进队和出队操作  
![]()  
```
from collections import deque
q = deque([1,2,3, 4, 5], 5)
q.append(6)  #队尾进队
q.popleft()  #队首出队
#用于双向队列
q.appendleft(1)  #队首进队
q.pop() #队尾出队
```
#### 栈和队列的应用--迷宫问题
![]()  
**栈--深度优先搜索**  
思路：从一个节点开始，任意找下一个能走的点，当找不到能走的点时，退回上一个点寻找是否有其他方向的点。  
使用栈存储当前路径  
```
maze = [
	[1,1,1,1,1,1,1,1,1,1],
	[1,0,0,1,0,0,0,1,0,1],
	[1,0,0,1,0,0,0,1,0,1],
    [1,0,0,0,0,1,1,0,0,1],
    [1,0,1,1,1,0,0,0,0,1],
    [1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,0,0,1,0,0,1],
    [1,0,1,1,1,0,1,1,0,1],
    [1,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1]
]
dirs = [
	lambda x, y: (x+1, y),
	lambda x, y: (x-1, y),
	lambda x, y: (x, y-1),
	lambda x, y: (x, y+1)
]

def maze_path(x1, y1, x2, y2):
    stack = []
    stack.append((x1, y1))
    #当栈空时表示没有路
    while(len(stack)>0):
        curNode = stack[-1]
        if curNode[0] == x2 and curNode[1] == y2: #走到终点了
            for p in stack:
                print(p)
            return True
        
        #x,y的四个方向，上：x-1,y 下：x+1, y 左：x,y-1 右：x, y+1
        for dir in dirs:
            nextNode = dir(curNode[0], curNode[1])
            #如果下一个节点能走
            if maze[nextNode[0], nextNode[1]] == 0:
                stack.append(nextNode)
                maze[nextNode[0], nextNode[1]] = 2 #2表示已经走过
        		break
            else:
                maze[nextNode[0], nextNode[1]] = 2
                stack.pop()
    else:
        print('没有路')
        return False
```
**队列--广度优先搜索**  
思路：从一个节点开始，寻找所有接下来继续走的点，继续不断寻找，直到找到出口。
使用队列存储当前正在考虑的节点  
```
from collections import deque

maze = [
	[1,1,1,1,1,1,1,1,1,1],
	[1,0,0,1,0,0,0,1,0,1],
	[1,0,0,1,0,0,0,1,0,1],
    [1,0,0,0,0,1,1,0,0,1],
    [1,0,1,1,1,0,0,0,0,1],
    [1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,0,0,1,0,0,1],
    [1,0,1,1,1,0,1,1,0,1],
    [1,1,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1]
]
dirs = [
	lambda x, y: (x+1, y),
	lambda x, y: (x-1, y),
	lambda x, y: (x, y-1),
	lambda x, y: (x, y+1)
]

def print_r(path):
    curNode = path[-1]
    realpath = []
    while curNode[2] != -1:
        realpath.append(curNode[0:2])
        curNode = path[curNode[2]]
    real.append(curNode[0:2])  #起点
    realpath.reverse()
    for node in realpath:
        print(node)
        
def maze_path_queue(x1, y1, x2, y2):
    queue = deque()
    queue.append((x1, y1, -1))
    path = []
    while len(queue) > 0:
        curNode = queue.pop()
        path.append(curNode)
        if curNode[0] == x2 and curNode[1] == y2:#终点
            print_r(path)
            return True
        for dir in dirs:
            nextNode = dir(curNode[0], curNode[1])
            if maze[nextNode[0]][nextNode[1]] == 0:
                queue.append((nextNode[0], nextNode[1], len(path) - 1))#后续节点进队，记录是哪个节点带来的
                maze[nextNode[0], nextNode[1]] = 2 #标记为已经走过
    else:
        print('没有路')
        return False
```
### 4.链表
链表是有一系列节点组成的元素集合。每个节点包含两个部分，数据域item和指向下一个节点的指针next.通过节点之间的相互连接，最终串联成一个链表。  
![]()  
```
#手动创建链表
class Node:
    def __init__(self, item):
        self.item = item
        self.next = None
a = Node(1)
b = Node(2)
c = Node(3)
a.next = b
b.next = c
print(a.next.next.item)  #3
```
#### 链表的创建
```
class Node:
    def __init__(self, item):
        self.item = item
        self.next = None
        
#头插法创建链表
def create_linklist_head(li):
    head = Node(li[0])
    for element in li[1:]:
        node = Node(element)
        node.next = head
        head = node
    return head

#尾插法创建链表
def create_linklist_tail(li):
    head = Node(li[0])
    tail = head
    for element in li[1:]:
        node = Node(element)
        tail.next = node
        tail = node
    return head

def print_linklist(lk):
    while lk:
        print(lk.item, end=',')
        lk = lk.next 
```
#### 链表的插入和删除
插入  
`p.next = curNode.next`  
`curNode.next = p`  
删除  
`p = curNode.next`  
`curNode.next = p.next`  
`del p`  
#### 双链表  
![]()  
![]() ![]() ![]() ![]()  
```
class Node(object):
    def __init__(self, item=None):
        self.item = item
        self.next = None
        self.prior = None
        
#插入p节点
#1
p.next = curNode
#2
curNode.next.prior = p
#3
p.prior = curNode
curNode.next = p
```
![]()  ![]()  ![]()  
```
#删除p节点
p = curNode.next
curNode.next = p.next
p.next.prior = curNode
del p
```
复杂度讨论
顺序表（列表/数组）与链表
1. 按元素值查找   O(n)         O(n)
2. 按下标查找     O(1)          O(n)
3. 在某元素后插入O(n)         O(1)
4. 删除某元素     O(n)          O(1)  

### 5.哈希表
哈希表是一个通过哈希函数来计算数据存储位置的数据结构，通常支持如下操作：  
insert(key, value):插入键值对(key, value)  
get(key):如果存在键为key的键值对则返回其value, 否则返回空值  
delete(key):删除键为key的键值对  
#### 直接寻址表
![]()  
直接寻址表缺点：
1. 当域U很大时，需要消耗大量内存，很不实际
2. 如果域U很大而实际出现的key很少，则大量空间被浪费
3. 无法处理关键字不是数字的情况  
#### 哈希
直接寻址表：key为k的元素放到k位置上  
改进直接寻址表: Hashing  
构建大小为m的寻址表T  
key为k的元素放到h(k)位置上  
h(k)是一个函数，其将域U映射表T[0, 1, ..., m-1]  
#### 哈希表
哈希表(Hash Table, 又称为散列表)，是一种线性表的存储结构。哈希表由一个直接寻址表和一个哈希函数组成。哈希函数h(k)将元素关键字k作为自变量，返回元素的存储下标。  
假设有一个长度为7的哈希表，哈希函数h(l)=k%7。元素集合{14, 22, 3, 5}的存储方式如下图。  
![]()  
##### 哈希冲突
哈希冲突：由于哈希表的大小是有限的，而要存储的值的总数量是无限的，因此对于任何哈希函数，都会出现两个不同元素映射到同一个位置的情况，这种情况叫做哈希冲突。  
比如：h(k)=k%7, h(0)=h(7)=h(14)...  
**解决哈希冲突--开放寻址法**  
**开放寻址法**：如果哈希函数返回的位置已经有值，则可以向后探查新的位置来存储这个值。  
1. 线性探查：如果位置i被占用，则探查i+1, i+2, ...
2. 二次探查：如果位置i被占用，则探查
3. 二度哈希：有n个哈希函数，当使用第1个哈希函数h1发生冲突时，则尝试使用h2, h3, ...

**拉链法**：哈希表每个位置都连接一个链表，当冲突发生时，冲突的元素将被加到该位置链表的最后。  
![]()  
```
class LinkList:
    class Node:
        def __init__(self, item=None):
            self.item = item
            self.next = None
            
    class LinkListIterator:
        def __init__(self, node):
            self.node = node
            
        def __next__(self):
            if self.node:
                cur_node = self.node
                self.node = cur_node.next
                return cur_node.item
            else:
                raise StopIteration
                
        def __iter__(self):
            return self
        
    def __init__(self, iterable=None): #iterable is list
        self.head = None
        self.tail = None
        if iterable:
            self.extend(iterable)
            
    def append(self, obj):
        s = LinkList.Node(obj)
        if not self.head:
            self.head = s
            self.tail = s
        else:
            self.tail.next = s
            self.tail = s
            
   	def extend(self, iterable):
        for obj in iterable:
            self.append(obj)
            
    def find(self, obj):
        for n in self:
            if n == obj:
                return True
            else:
                return False
    
    def __iter__(self):
        return self.LinkListIterator(self.head)
    
    def __repr__(self):
        return "<<"+",".join(map(str, self)+">>")
    
#类似于集合的结构,所以不能有重复的元素
class HashTable:
    def __init__(self, size=101):
        self.size = size
        self.T = [LinkList() for i in range(self.size)]
        
    def h(self, k):
        return k % self.size
   	
    def insert(self, k):
        i = self.h(k)
        if self.find(k):
            print('Duplicated Insert')
        else:
            self.T[i].append(k)
            
       
   	def find(self, k):
        i = self.h(k)
        return self.T[i].find(k)

ht = HashTable()
ht.insert(0)
ht.insert(1)
ht.insert(3)
ht.insert(102)
ht.insert(508)

print(",".join(map(str, ht.T)))
```
#### 哈希表的应用--集合与字典
1.字典和集合都是通过哈希表来实现的  
a ={"name": "Alex", "age": 18, "gender": "Man"}  
2.使用哈希表存储的字典，通过哈希函数将字典的键映射为下标。假设h('name')=3, h('age')=1, h('gender')=4, 则哈希表存储为[None, 18, None, 'Alex', 'Man']  
3.如果发生哈希冲突，则通过拉链法或开放寻址法解决。