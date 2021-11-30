#输入描述：
#多组输入每组输入 1 个正整数 n 。( n 不大于 30000 )
#输出描述：
#不大于n的与7有关的数字个数，例如输入20，与7有关的数字包括7,14,17.

while True:
    try:
        num = int(input())
        count = 0
        for i in range(1,num+1):
            if i % 7 == 0 or '7' in str(i):
                count += 1
        print(count)
    except:
        break