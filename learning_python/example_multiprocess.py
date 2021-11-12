import time
from PIL import Image, ImageFilter
import concurrent.futures

img_names = [
    'photo-1549692520-acc6669e2f0c.jpg',
    'photo-1550439062-609e1531270e.jpg',
    'photo-1516117172878-fd2c41f4a759.jpg',
    'photo-1532009324734-20a7a5813719.jpg',
    'photo-1524429656589-6633a470097c.jpg',
    'photo-1530224264768-7ff8c1789d79.jpg',
    'photo-1564135624576-c5c88640f235.jpg'
]

t1 = time.perf_counter()

size = (1200, 1200)
'''
for img_name in img_names:
    img = Image.open(img_name)

    img = img.filter(ImageFilter.GaussianBlur(15))

    img.thumbnail(size)
    img.save(f'processed/{img_name}')
    print(f'{img_name} was processed...')
'''

def process_image(img_name):
    img = Image.open(img_name)
    img = img.filter(ImageFilter.GaussianBlur(15))
    img.thumbnail(size)
    img.save(f'processed/{img_name}')
    print(f'{img_name} was processed...')

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_image, img_names)
    
t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')
