from struct import unpack
import os
from tqdm import tqdm

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        


bads = []

root_dir = "./data/image/other"
images = os.listdir(root_dir)
print(f"Total images - {len(images)}")
for img in tqdm(images):
  image_path = os.path.join(root_dir,img)
  image = JPEG(image_path) 
  try:
    image.decode()   
  except:
    bads.append(image_path)

# print(bads)
print(f"Total bad images - {len(bads)}")
print('Removing bad image')
for img_path in tqdm(bads):
  os.remove(img_path)