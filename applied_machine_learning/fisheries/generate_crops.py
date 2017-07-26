import os

import numpy as np
import pandas as pd
import scipy.misc
import skimage.draw
from skimage.io import imsave

import utils

# TRAIN_DIR = 'data/train_split'
# OUT_DIR = 'data/train_crops'
TRAIN_DIR = 'data/val_split'
OUT_DIR = 'data/val_crops'
LABELS_DIR = 'bbox'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
MAX_EXAMPLES = 10000
OUT_SIZE = 280
NET_SIZE = 224
EXPAND_BBOX = 0.5


os.makedirs(OUT_DIR, exist_ok=True)
for fish_class in FISH_CLASSES:
  print('processing ' + fish_class)
  labels = pd.read_json(LABELS_DIR + '/' + fish_class.lower() + '_labels.json')
  os.makedirs(OUT_DIR + '/' + fish_class, exist_ok=True)
  filenames = []
  i_examples = 0
  for _, image in labels.iterrows():
    # check if file exits and is annotated
    filename = TRAIN_DIR + '/' + fish_class + '/' + image.filename
    if not os.path.isfile(filename):
      print('missing ' + filename)
    else:
      img = scipy.misc.imread(filename, mode='RGB')
      img_size = np.array(img.shape[:2])
      for i, anno in enumerate(image.annotations):
        bbox = utils.anno2bbox(anno)
        bbox = utils.expand_bbox(bbox, EXPAND_BBOX, max_limits=img_size)
        # TODO: add margin for jittering
        bbox = bbox.astype(np.int32)

        # crop and resize
        img_bbox = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        out_scale = OUT_SIZE / max(img_bbox.shape[0:2])
        img_bbox = scipy.misc.imresize(img_bbox, out_scale)

        imsave(OUT_DIR + '/' + fish_class + '/' + os.path.splitext(image.filename)[0] + '-' + str(i) + '.jpg', img_bbox)

      i_examples += 1
      if i_examples > MAX_EXAMPLES:
        break
