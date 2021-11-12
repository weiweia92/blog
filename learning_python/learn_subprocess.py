import subprocess

#subprocess.run('ls -la', shell=True)
#p1 = subprocess.run(['ls','-la', 'dne'], stderr=subprocess.DEVNULL)
file ='test.txt'
p1 = subprocess.run(['cat', file], capture_output=True, text=True)
#with open('output.txt', 'w') as f:
#    p1 = subprocess.run(['ls','-la'], stdout=f, text=True)
p3 = subprocess.run('cat test.txt | grep -n test', shell=True, capture_output=True, text=True)
#print(p1)
#print(p1.args)
#print(p1.returncode)  # 0 means successfully(zero errors)
#print(p1.returncode) #2 means error
#print(p1.stderr)
print(p1.stdout)


p2 = subprocess.run(['grep', '-n', 'test'],
                    capture_output=True, text=True, input=p1.stdout)

print(p2.stdout)
print(p3.stdout)
