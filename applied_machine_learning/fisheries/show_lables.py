import os

import numpy as np
import pandas as pd
import scipy.misc
import skimage.draw
from skimage.data import imread
from skimage.draw._draw import _coords_inside_image
from skimage.io import imsave

TRAIN_DIR = 'data/train'
OUT_DIR = 'data/train_labeled'
LABELS_DIR = 'labels'
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
    if os.path.isfile(filename) and len(image.annotations) == 2:
      img = scipy.misc.imread(filename, mode='RGB')

      annotation = image.annotations[0]
      coords = skimage.draw.circle(annotation['y'], annotation['x'], 10)
      skimage.draw.set_color(img, coords, [0, 255, 0])

      annotation = image.annotations[1]
      coords = skimage.draw.circle(annotation['y'], annotation['x'], 10)
      skimage.draw.set_color(img, coords, [255, 0, 0])

      size = np.array(img.shape[:2])
      scale = MAX_SIZE / np.max(size)
      new_size = np.floor(size * scale).astype(np.int32)
      img = scipy.misc.imresize(img, (new_size[0], new_size[1]))

      imsave(OUT_DIR + '/' + fish_class + '/' + image.filename, img)

      filenames.append(image.filename)

      i_examples += 1
      if i_examples > MAX_EXAMPLES:
        break

  with open(OUT_DIR + '/' + fish_class + '.txt', 'w') as text_file:
    text_file.write('\n'.join(filenames))
