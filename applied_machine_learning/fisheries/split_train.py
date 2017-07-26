import math
import os
import random

from sklearn.model_selection import train_test_split

FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

INPUT_DIR = 'data/train'
TRAIN_DIR = 'data/train_split'
VAL_DIR = 'data/val_split'

VAL_SIZE = 0.2


filepaths = []
for fish_class in FISH_CLASSES:
  filepaths.extend([fish_class + '/' + filename for filename in
                   os.listdir(INPUT_DIR + '/' + fish_class)])

random.shuffle(filepaths)
ind = math.ceil(len(filepaths) * 0.2)
filepaths_val = filepaths[:ind]
filepaths_train = filepaths[ind:]

def make_symlinks(directory, filepaths):
  os.makedirs(directory, exist_ok=True)
  for fish_class in FISH_CLASSES:
    os.makedirs(directory + '/' + fish_class, exist_ok=True)
  for filepath in filepaths:
    src = os.path.abspath(directory + '/' + filepath)
    dst = os.path.abspath(INPUT_DIR + '/' + filepath)
    os.symlink(dst, src)

make_symlinks(TRAIN_DIR, filepaths_train)
make_symlinks(VAL_DIR, filepaths_val)
