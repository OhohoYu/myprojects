import os

import numpy as np
import pandas as pd
import scipy.misc
from sklearn.cluster import KMeans

import utils

TRAIN_DIR = 'data/train'
OUT_DIR = 'data'
LABELS_DIR = 'bbox'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
MAX_EXAMPLES = 10000
EXPAND_BBOX = 0.5

N_SAMPLES = 100


os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isfile(OUT_DIR + '/all_bboxes.npy'):
  bboxes = []
  for fish_class in FISH_CLASSES:
    print('processing ' + fish_class)
    labels = pd.read_json(LABELS_DIR + '/' + fish_class.lower() + '_labels.json')
    for _, image in labels.iterrows():
      # check if file exits and is annotated
      filename = TRAIN_DIR + '/' + fish_class + '/' + image.filename
      if not os.path.isfile(filename):
        print('missing ' + filename)
      else:
        img = scipy.misc.imread(filename, mode='RGB')
        img_size = np.array(img.shape[:2])
        for anno in image.annotations:
          bbox = utils.anno2bbox(anno)
          bbox = utils.expand_bbox(bbox, EXPAND_BBOX, max_limits=img_size)
          # normalize
          bbox[[0,2]] = bbox[[0,2]] / img_size[0]
          bbox[[1,3]] = bbox[[1,3]] / img_size[1]
          bboxes.append(bbox)
  bboxes = np.array(bboxes, dtype=np.float32)
  np.save(OUT_DIR + '/all_bboxes.npy', bboxes)
else:
  bboxes = np.load(OUT_DIR + '/all_bboxes.npy')

# cluster
kmeans = KMeans(n_clusters=N_SAMPLES, random_state=0).fit(bboxes)
proposals = kmeans.cluster_centers_

np.save(OUT_DIR + '/proposals.npy', proposals)
