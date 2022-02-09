import os
import signal
import subprocess
import time

proc = subprocess.Popen('/Users/leexuewei/blog/TTS/Speech-Zone/Tacotron2/Tacotron2/atexit_signal_child.py')
print('PARENT: Pausing before sending signal...')
time.sleep(1)
print('PARENT: Signaling child')
os.kill(proc.pid, signal.SIGTERM)