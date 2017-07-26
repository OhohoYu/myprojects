import os
import pickle

import numpy as np
import pandas as pd
import scipy.misc
import skimage.draw
from skimage.data import imread
from skimage.draw._draw import _coords_inside_image
from skimage.io import imsave

MODEL_NAME = 'locate_fish_only_vgg16'

OUT_DIR = 'data/test_pos_vis'
# OUT_DIR = 'data/train_pos_vis'
# OUT_DIR = 'data/val_pos_vis'
IMGS_DIR = 'data/test_stg1'
# IMGS_DIR = ''
PREDS_DIR = 'data/' + MODEL_NAME
MAX_SIZE = 256

test_preds = np.load(PREDS_DIR + '/test_preds.npy')
filenames = pickle.load(open(PREDS_DIR + '/test_preds_files.p', 'rb'))

os.makedirs(OUT_DIR, exist_ok=True)

for i, (filename, label) in enumerate(zip(filenames, test_preds)):
  # check if file exits and is annotated
  imgs_dir = IMGS_DIR
  if imgs_dir:
    imgs_dir += '/'
  filepath = imgs_dir + filename
  img = scipy.misc.imread(filepath, mode='RGB')

  size = np.array(img.shape[:2])
  scale = MAX_SIZE / np.max(size)
  new_size = np.floor(size * scale).astype(np.int32)
  img = scipy.misc.imresize(img, (new_size[0], new_size[1]))

  # clamp labels
  label = np.minimum(np.maximum(label, 0), 1)
  # and resize them
  label[0] *= new_size[0]
  label[1] *= new_size[1]

  coords = skimage.draw.circle(label[0], label[1], 10)
  skimage.draw.set_color(img, coords, [0, 255, 0])

  fileout = OUT_DIR + '/' + filename
  os.makedirs(os.path.dirname(fileout), exist_ok=True)
  imsave(fileout, img)
