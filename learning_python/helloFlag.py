from absl import app
from absl import flags
import os
 
FLAGS = flags.FLAGS
 
flags.DEFINE_string('gpu', None, 'comma separated list of GPU(s) to use.')
 
 
def main(argv):
    del argv
    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    else:
        print('Please assign GPUs.')
        exit()
 
 
if __name__ == '__main__':
    app.run(main)

# python helloFlag.py --gpu 0
