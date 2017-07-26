import os

import numpy as np
import pandas as pd
import scipy.misc
import skimage.draw
from skimage.io import imsave

import utils

IMG_DIR = 'data/train_split/NoF'
OUT_DIR = 'data/train_crops/NoF'
# IMG_DIR = 'data/val_split/NoF'
# OUT_DIR = 'data/val_crops/NoF'
N_SAMPLES = 80

# IMG_DIR = 'data/test_stg1'
# OUT_DIR = 'data/test_crops'
# N_SAMPLES = 100

PROPOSALS = 'data/proposals.npy'

SAMPLE = False
OUT_SIZE = 280
NET_SIZE = 224

proposals = np.load(PROPOSALS)
n_proposals = proposals.shape[0]

os.makedirs(OUT_DIR, exist_ok=True)
filenames = os.listdir(IMG_DIR)
for i, filename in enumerate(filenames):
  if i%100 == 0: print(str(i) + '/' + str(len(filenames)))

  img = scipy.misc.imread(IMG_DIR + '/' + filename, mode='RGB')
  img_size = np.array(img.shape[:2])

  if SAMPLE:
    indices = np.arange(n_proposals)
    np.random.shuffle(indices)
    proposals = proposals[indices]
  for j in range(min(n_proposals, N_SAMPLES)):
    bbox = proposals[j].copy()
    # TODO: add margin for jittering
    # denormalize
    bbox[[0,2]] = bbox[[0,2]] * img_size[0]
    bbox[[1,3]] = bbox[[1,3]] * img_size[1]
    bbox = bbox.astype(np.int32)

    # crop and resize
    img_bbox = img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
    out_scale = OUT_SIZE / max(img_bbox.shape[0:2])
    img_bbox = scipy.misc.imresize(img_bbox, out_scale)

    imsave(OUT_DIR + '/' + os.path.splitext(filename)[0] + '-' + str(j) + '.jpg', img_bbox)
