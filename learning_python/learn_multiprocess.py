import multiprocessing
import time
import concurrent.futures

start = time.perf_counter()


def do_something(second):
    print(f'Sleeping {second} seconds...')
    time.sleep(second)
    #print('Done Sleeping...')
    return 'Done Sleeping...'
#do_something()
#do_something()
####################################################
#create two process but not run.
#(1)p1 = multiprocessing.Process(target=do_something)
#(1)p2 = multiprocessing.Process(target=do_something)

#(1)p1.start()
#(1)p2.start()

#(1)p1.join()
#(1)p2.join()
###################################################

with concurrent.futures.ProcessPoolExecutor() as executor:
    secs = [5, 4, 3, 2, 1]
    results = executor.map(do_something, secs)

    for result in results:
        print(result)
    #f1 = executor.submit(do_something, 1)
    #print(f1.result())

    #(3)secs = [5, 4, 3, 2, 1]
    #iterator
    #(3)results = [executor.submit(do_something, sec) for sec in secs]

    #(3)for f in concurrent.futures.as_completed(results):
    #(3)    print(f.result())



#(2)processes = []

#(2)for _ in range(10):
#(2)    p = multiprocessing.Process(target=do_something, args=[1.5])
#(2)    p.start()
#(2)    processes.append(p)


#(2)for process in processes:
#(2)    process.join()
    
finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
