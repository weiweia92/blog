import itertools as it
import ipdb

def is_brother_word(L):
    res = it.permutations(L)
    brother_words = []
    for item in res:
        brother_word = ''
        for i in item:
            brother_word += i
        brother_words.append(brother_word)
    return brother_words

def strs2list(strings):
    L = []
    for i in strings:
        L.append(i)
    return L

def intersection(l1,l2):
    intersec = []
    L = min(len(l1), len(l2))
    for i in range(L):
        if l1[i] in l2:
            intersec.append(l1[i])
    return intersec

while True:
    try:
        line = input().split(' ')
        n = int(line[0])
        words = line[1:-2]
        target = line[-2]
        target_list = strs2list(target)
        brother_words = is_brother_word(target_list)
        brother_words.remove(target)
        sorted_brother_words = sorted(brother_words)
        id = int(line[-1])
        print(len(intersection(words, sorted_brother_words)))
        L = intersection(words, sorted_brother_words)
        if L[id-1]:
            print(L[id-1])
    except:
        break
        