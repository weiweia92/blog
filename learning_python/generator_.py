import memory_profiler as mem_profile
import random
import time

names = ['John', 'Corey', 'Adam', 'Steve', 'Rick', 'Thomas']
majors = ['Math', 'Engineering', 'CompSci', 'Arts', 'Business']

print(f'Memory (Before):{mem_profile.memory_usage()}')

def people_list(num_people):
    result = []
    for i in range(num_people):
        person = {'id': i,
                  'name': random.choice(names),
                  'major': random.choice(majors)}
        result.append(person)
    return result

def people_generator(num_people):
    for i in range(num_people):
        person = {'id': i,
                  'name': random.choice(names),
                  'major': random.choice(majors)}
        yield person

t1 = time.perf_counter()
people = people_list(1000000)
t2 = time.perf_counter()
print(f'Memory (After):{mem_profile.memory_usage()}')
print('Took {} seconds'.format(t2-t1))

print(f'Memory (Before):{mem_profile.memory_usage()}')


t3 = time.perf_counter()
people1 = people_generator(1000000)
t4 = time.perf_counter()

print(f'Memory (After):{mem_profile.memory_usage()}')
print('Took {} seconds'.format(t4-t3))


def square_numbers(nums):
    for i in nums:
        yield (i * i) # yield change it to generator.

my_nums = square_numbers([1, 2, 3, 4, 5])
#<generator object square_numbers at 0x7f4899896550>
print(next(my_nums))
print(next(my_nums))
print(next(my_nums))
print(next(my_nums))
print(next(my_nums))

print(my_nums)

my_nums = square_numbers([1, 2, 3, 4, 5])
for num in my_nums:
    print(num)


