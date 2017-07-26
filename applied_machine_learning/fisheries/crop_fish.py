import os
import pickle

import numpy as np
import pandas as pd
import scipy.misc
import skimage.draw
from skimage.data import imread
from skimage.draw._draw import _coords_inside_image
from skimage.io import imsave

MODEL_NAME = 'locate_fish_vgg16'

# OUT_DIR = 'data/test_crop'
OUT_DIR = 'data/train_crop'
# IMGS_DIR = 'data/test_stg1'
IMGS_DIR = 'data/train'
PREDS_DIR = 'data/' + MODEL_NAME
# PREDS_PREFIX = 'test'
# PREDS_PREFIX = 'train'
PREDS_PREFIX = 'train'
MAX_SIZE = 256

def clamp(x, min_value, max_value):
  return np.minimum(np.maximum(x, min_value), max_value)

def crop(im, row0, col0, n_rows, n_cols):
  img_crop = np.zeros((new_size, new_size, 3), dtype=im.dtype)

  row1 = row0 + new_size
  col1 = col0 + new_size

  row0_ = clamp(row0, 0, size[0])
  row1_ = clamp(row1, 0, size[0])
  col0_ = clamp(col0, 0, size[1])
  col1_ = clamp(col1, 0, size[1])
  row0_delta = row0_ - row0
  row1_delta = row1_ - row1
  col0_delta = col0_ - col0
  col1_delta = col1_ - col1

  img_crop[row0_delta:new_size+row1_delta, col0_delta:new_size+col1_delta, :] = im[row0_:row1_, col0_:col1_, :]

  return img_crop



test_preds = np.load(PREDS_DIR + '/' + PREDS_PREFIX + '_preds.npy')
filenames = pickle.load(open(PREDS_DIR + '/' + PREDS_PREFIX + '_preds_files.p', 'rb'))

os.makedirs(OUT_DIR, exist_ok=True)

for i, (filename, label) in enumerate(zip(filenames, test_preds)):
  # check if file exits and is annotated
  imgs_dir = IMGS_DIR
  if imgs_dir:
    imgs_dir += '/'
  filepath = imgs_dir + filename
  img = scipy.misc.imread(filepath, mode='RGB')

  size = np.array(img.shape[:2])

  # clamp labels
  label = clamp(label, 0, 1)
  # and resize them
  label[[0,2]] *= size[0]
  label[[1,3]] *= size[1]
  # take centre
  centre_row = np.mean(label[[0,2]]).astype('int32')
  centre_col = np.mean(label[[1,3]]).astype('int32')

  new_size = (np.min(size) * 0.7).astype('int32')
  row0 = (centre_row - new_size / 2).astype('int32')
  col0 = (centre_col - new_size / 2).astype('int32')
  img = crop(img, row0, col0, new_size, new_size)

  fileout = OUT_DIR + '/' + filename
  os.makedirs(os.path.dirname(fileout), exist_ok=True)
  imsave(fileout, img)
