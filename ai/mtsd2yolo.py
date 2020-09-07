import argparse
import os
import csv
from PIL import Image
from shutil import copy

parser = argparse.ArgumentParser(description='Prepare MTSD (Detection) in YOLOv4 (Darknet) dataset format')
parser.add_argument(
    '--mtsd',
    default='MTSD',
    type=str,
    dest='mtsd_dir',
    help='path to MTSD directory'
)
parser.add_argument(
    '-d', '--data',
    default='data',
    type=str,
    dest='data_dir',
    help='path to YOLOv4 data directory'
)
parser.add_argument(
    '-c', '--class',
    default='class_name.csv',
    type=str,
    dest='class_name',
    help='path to class.csv'
)
args = parser.parse_args()

labels = []

# MTSD
gt_txt = 'GT.txt'
detection_dir = os.path.join(args.mtsd_dir, 'Detection')

# YOLOv4
data_dir = args.data_dir
obj_dir = os.path.join(data_dir, 'obj')
if (not os.path.exists(obj_dir)):
    os.makedirs(obj_dir, exist_ok=True)
obj_names = 'obj.names'
obj_data = 'obj.data'
train_txt = 'train.txt'
valid_txt = 'valid.txt'

with open(args.class_name) as f:
    reader = csv.reader(f)
    class_name = {row[0]: row[1] for row in reader}

for f in os.listdir(obj_dir):
    if f.endswith('.txt'):
        os.remove(os.path.join(obj_dir, f))

class_names = []
class_count = {}
images = {}

with open(os.path.join(args.mtsd_dir, gt_txt)) as f:
    reader = csv.reader(f, delimiter=';')
    i = 0
    for row in reader:
        if i != 0:
            # row: File Name;X;Y;Width;Height;Sign Type;Sign Group;Sign Class;TS Class;Class ID;TS Color;Shape;Shape ID;Lightning;Image Source
            img_name = row[0].replace("'", '')
            img_path = os.path.join(detection_dir, img_name)
            if img_name not in images:
                img_width, img_height = Image.open(img_path).size
                images[img_name] = [img_width, img_height]
                copy(img_path, obj_dir)
                print('[copy]', end =' ')
            else:
                img_width = images[img_name][0]
                img_height = images[img_name][1]
                print('      ', end =' ')

            name = class_name[row[9]]
            if name not in class_names:
                class_names.append(name)
                class_count[name] = 1
            else:
                class_count[name] += 1
            class_index = class_names.index(name)

            x = (int(row[1]) + int(row[3])/2) / img_width
            y = (int(row[2]) + int(row[4])/2) / img_height
            w = int(row[3]) / img_width
            h = int(row[4]) / img_height

            txt_name = os.path.splitext(img_name)[0] + '.txt'
            with open(os.path.join(obj_dir, txt_name), 'a') as txt:
                txt.write(f'{class_index} {x} {y} {w} {h}\n')

            print(f'({i}) {img_name} {class_index} {x} {y} {w} {h}')

        i += 1

classes = len(class_names)

with open(os.path.join(args.data_dir, obj_data), 'w') as f:
    f.write(f'classes = {classes}\n'
            f'train = {os.path.join("data", train_txt)}\n'
            f'valid = {os.path.join("data", valid_txt)}\n'
            f'names = {os.path.join("data", obj_names)}\n'
            f'backup = {os.path.join("backup", "")}')

with open(os.path.join(args.data_dir, obj_names), 'w') as f:
    for name in class_names:
        f.write(f'{name}\n')

with open(os.path.join(args.data_dir, train_txt), 'w') as f:
    for image in images:
        f.write(f'{os.path.join("data", "obj", image)}\n')

copy(os.path.join(args.data_dir, train_txt), os.path.join(args.data_dir, valid_txt))

print(f'\nProcessed total {classes} classes:\n{class_names}')
print(f'\nClass frequency:\n{class_count}')
print(f'\n{obj_data} {obj_names} {train_txt} {valid_txt} created')
