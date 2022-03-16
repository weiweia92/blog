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
            if list1.val < list2.val:
                curr.next = list1
                list1 = list1.next
            curr = curr.next
        curr.next = list1 or list2
        return dummy.next