import os

import numpy as np
import pandas as pd
import scipy.misc
import skimage.draw
from skimage.io import imsave

TRAIN_DIR = 'data/train'
OUT_DIR = 'data/train_bbox'
LABELS_DIR = 'bbox'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
MAX_EXAMPLES = 10000
MAX_SIZE = 256

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
      for i, anno in enumerate(image.annotations):
        img_bbox = img.copy()
        coords = skimage.draw.circle(anno['y'], anno['x'], 10)
        skimage.draw.set_color(img_bbox, coords, [255, 0, 0])

        coords = skimage.draw.circle(anno['y'], anno['x'] + anno['width'], 10)
        skimage.draw.set_color(img_bbox, coords, [0, 255, 0])

        coords = skimage.draw.circle(anno['y'] + anno['height'], anno['x'] + anno['width'], 10)
        skimage.draw.set_color(img_bbox, coords, [0, 0, 255])

        coords = skimage.draw.circle(anno['y'] + anno['height'], anno['x'], 10)
        skimage.draw.set_color(img_bbox, coords, [0, 255, 255])

        size = np.array(img.shape[:2])
        scale = MAX_SIZE / np.max(size)
        new_size = np.floor(size * scale).astype(np.int32)
        img_bbox = scipy.misc.imresize(img_bbox, (new_size[0], new_size[1]))

        imsave(OUT_DIR + '/' + fish_class + '/' + os.path.splitext(image.filename)[0] + '-' + str(i) + '.jpg', img_bbox)

      i_examples += 1
      if i_examples > MAX_EXAMPLES:
        break
