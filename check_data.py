from os import listdir
from PIL import Image
import pathlib

IMAGE_DIR = "'./data/image/other'"

count = 0
for filename in listdir():
  if filename.endswith(('.jpg','.jpeg''.png')):
    try:
      img = Image.open(IMAGE_DIR+filename) # open the image file
      img.verify() # verify that it is, in fact an image
    except (IOError, SyntaxError) as e:
      print('Bad file:', filename, e) # print out the names of corrupt files
      count = count + 1

print(f"Total bad file - {count}")

data_dir = pathlib.Path("./data/image/")
image_count = len(list(data_dir.glob('*/*.jpg')))

print(f"image dir - {data_dir}")
print(f"image count - {image_count}")
