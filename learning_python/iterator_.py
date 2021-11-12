# list is iterable but not a iterator.

nums = [1, 2, 3]

i_nums = nums.__iter__() # result is the same with 'i_nums = iter(nums)'

#print(dir(nums)) # inner method
#list does not have __next__, so it is not a iterator.but it has __iter__,so it has iterable.
#['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']
print(i_nums) #<list_iterator object at 0x7fb985361b90>
print(next(i_nums))

while True:
    try:
        item = next(i_nums)
        print(item)

    except StopIteration:
        break
    
print(dir(i_nums))
# iterator is iterable! so it has __iter__ method
#['__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__length_hint__', '__lt__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__setstate__', '__sizeof__', '__str__', '__subclasshook__']


class MyRange():

    def __init__(self, start, end):
        self.value = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.value >= self.end:
            raise StopIteration
        current = self.value
        self.value += 1
        return current

nums = MyRange(1, 10)

for num in nums:
    print(num)

print(next(nums))

