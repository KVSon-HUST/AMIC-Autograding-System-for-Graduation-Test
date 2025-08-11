from PIL import Image
import os

source_folder = 'E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Processed_data/test2/'
destination_folder = 'E:/OneDrive - Hanoi University of Science and Technology/HUST/Competition/AMIC/Processed_data/test2_changedsize/'
directory = os.listdir(source_folder)
print(directory)

for item in directory:
    img = Image.open(source_folder + item)
    width, height = img.size
    ratio = width / height
    new_width = 640
    new_height = int(new_width/ ratio)
    imgResized = img.resize((new_width, new_height))
    imgResized.save(destination_folder + item[:-4] + '.jpg', quality = 100)