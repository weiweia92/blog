import atexit

def all_done():
    print('all_done()')
print('Registering')
atexit.register(all_done)
print('Registered')