import time
#import threading
import concurrent.futures


start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    #print('Done Sleeping...')
    return f'Done Sleeping...{seconds}'

#(0)do_something()
#(0)do_something()

with concurrent.futures.ThreadPoolExecutor() as executor:
    #f1 = executor.submit(do_something, 1)
    secs = [5, 4, 3, 2, 1]
    #results is iterator.
    #(4)results = [executor.submit(do_something, sec) for sec in secs]
    results = executor.map(do_something, secs)

    for result in results:
        print(result)
    
    #(4)for f in concurrent.futures.as_completed(results):
    #(4)    print(f.result())
    #print(f1.result())
'''
# 10 times
(3)threads = []

(3)for _ in range(10):
(3)    t = threading.Thread(target=do_something, args=[1.5])
(3)    t.start()
(3)    threads.append(t)

(3)for thread in threads:
(3)    thread.join()
'''    
#######################################################
#(1)t1 = threading.Thread(target=do_something)
#(1)t2 = threading.Thread(target=do_something)

#run thread
#(1)t1.start()
#(1)t2.start()

#(1)t1.join()
#(1)t2.join()
########################################################

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')



