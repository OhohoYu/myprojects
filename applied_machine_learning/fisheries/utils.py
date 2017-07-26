import numpy as np


def clamp(x, min_value, max_value):
  return np.minimum(np.maximum(x, min_value), max_value)


def compute_intersection(a, b):
  dy = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
  dx = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
  return dx * dy


def compute_area(a):
  return (a[2] - a[0]) * (a[3] - a[1])


def compute_iou(a, b):
  dy = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
  dx = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
  inter = dx * dy

  a_area = (a[2] - a[0]) * (a[3] - a[1])
  b_area = (b[2] - b[0]) * (b[3] - b[1])

  iou = inter / (a_area + b_area - inter)

  return iou


def expand_bbox(bbox, scale, max_limits=None):
  h = (bbox[2] - bbox[0])
  w = (bbox[3] - bbox[1])
  dy = h*2 * scale * (1 / (1 + h / w))
  dx = w*2 * scale * (1 / (1 + w / h))
  bbox[0] -= dy
  bbox[1] -= dx
  bbox[2] += dy
  bbox[3] += dx

  if max_limits is not None:
    bbox[[0,2]] = clamp(bbox[[0,2]], 0, max_limits[0])
    bbox[[1,3]] = clamp(bbox[[1,3]], 0, max_limits[1])

  return bbox


def anno2bbox(anno):
  bbox = np.ndarray((4))
  bbox[0] = anno['y']
  bbox[1] = anno['x']
  bbox[2] = anno['y'] + anno['height']
  bbox[3] = anno['x'] + anno['width']
  return bbox
