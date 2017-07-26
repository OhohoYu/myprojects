import datetime
import os
import os.path
import random

import numpy as np
import pandas as pd
import scipy.misc
import skimage.io
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from image import ImageDataGenerator
from keras import backend as K
from keras import metrics
# from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import (Activation, Convolution2D, Dense, Dropout, Flatten,
                          GlobalMaxPooling2D, MaxPooling2D, ZeroPadding2D)
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

# from keras.applications.inception_v3 import InceptionV3



MODEL_NAME = 'crops_resnet50-5'

TRAIN_DIR = 'data/train_crops'
VAL_DIR = 'data/val_crops'
TEST_DIR = 'data/test_crops'

FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

TEMP_DIR = '/dev/shm/' + MODEL_NAME

ROWS = 224
COLS = 224
CHANNELS = 3
N_PROPOSALS = 100

BEST_WEIGHTS = '0.02.10'


def get_n_samples(directory):
  n = 0
  for fish_class in FISH_CLASSES:
    n += len(os.listdir(directory + '/' + fish_class))
  return n


def preprocess_input(x, dim_ordering='default'):
  """Preprocesses a tensor encoding a batch of images.
  # Arguments
    x: input Numpy tensor, 4D.
    dim_ordering: data format of the image tensor.
  # Returns
    Preprocessed tensor.
  """
  if dim_ordering == 'default':
    dim_ordering = K.image_dim_ordering()
  assert dim_ordering in {'tf', 'th'}

  if dim_ordering == 'th':
    # 'RGB'->'BGR'
    x = x[::-1, :, :]
    # Zero-center by mean pixel
    x[0, :, :] -= 103.939
    x[1, :, :] -= 116.779
    x[2, :, :] -= 123.68
  else:
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
  return x



def get_image_paths(fish):
  """Load files from train folder"""
  fish_dir = TRAIN_DIR+'{}'.format(fish)
  images = [fish+'/'+im for im in os.listdir(fish_dir)]
  return images


def read_image(src):
  """Read and resize individual images"""
  im = scipy.misc.imread(src, mode='RGB')
  im = scipy.misc.imresize(im, (ROWS, COLS))
  return im


def pop_layer(model):
  if not model.outputs:
    raise Exception('Sequential model cannot be popped: model is empty.')

  model.layers.pop()
  if not model.layers:
    model.outputs = []
    model.inbound_nodes = []
    model.outbound_nodes = []
  else:
    model.layers[-1].outbound_nodes = []
    model.outputs = [model.layers[-1].output]
  model.built = False
  return model


def load_train_data():
  files = []
  y_all = []
  for fish in FISH_CLASSES:
    fish_files = get_image_paths(fish)
    files.extend(fish_files)
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
  y_all = np.array(y_all)

  if not os.path.isfile(TEMP_DIR + '/X_all.npy'):
    X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.float32)
    for i, im in enumerate(files):
      X_all[i] = read_image(TRAIN_DIR + im)
      if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))
    X_all = preprocess_input(X_all)
    print('saving X_all')
    np.save(TEMP_DIR + '/X_all.npy', X_all)
  else:
    X_all = np.load(TEMP_DIR + '/X_all.npy')
  print(X_all.shape)

  # One Hot Encoding Labels
  y_all = LabelEncoder().fit_transform(y_all)
  y_all = np_utils.to_categorical(y_all)

  X_train, X_valid, y_train, y_valid = train_test_split(
    X_all, y_all, test_size=0.1, random_state=23)

  return X_train, X_valid, y_train, y_valid


def create_model():
  # create the base pre-trained model
  # base_model = VGG16(weights='imagenet', include_top=True)
  # base_model = InceptionV3(weights='imagenet', include_top=True)
  base_model = ResNet50(weights='imagenet', include_top=True)

  # remove last the classification layer
  x = base_model.layers[-2].output
  predictions = Dense(len(FISH_CLASSES), activation='softmax', name='predictions2')(x)

  # this is the model we will train
  model = Model(input=base_model.input, output=predictions)
  return model


def train():
  os.makedirs(TEMP_DIR + '/models', exist_ok=True)

  model = create_model()

  batch_size=64
  nb_epoch=100

  n_train_samples = get_n_samples(TRAIN_DIR)
  n_val_samples = get_n_samples(VAL_DIR)

  model.load_weights('/dev/shm/' + MODEL_NAME + '/models/weights.' + BEST_WEIGHTS + '.hdf5')
  model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=1e-4),
                loss='categorical_crossentropy',
                metrics=[metrics.categorical_accuracy])

  early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

  # saves the model weights after each epoch if the validation loss decreased
  os.makedirs('models/' + MODEL_NAME, exist_ok=True)
  checkpointer = ModelCheckpoint(
    filepath=TEMP_DIR + '/models/weights.{val_loss:.2f}.{epoch:02d}.hdf5', verbose=1,
    save_best_only=True, save_weights_only=True)

  os.makedirs('tf_logs', exist_ok=True)
  tensorboard = TensorBoard(log_dir='./tf_logs', histogram_freq=0,
                            write_graph=False, write_images=False)


  traingen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=True,
                                zoom_range=0.2)

  valgen = ImageDataGenerator(preprocessing_function=preprocess_input)

  train_flow = traingen.flow_from_directory(TRAIN_DIR,
                                           target_size=(ROWS, COLS),
                                           classes=FISH_CLASSES,
                                           batch_size=batch_size,
                                           shuffle=True)

  val_flow = valgen.flow_from_directory(VAL_DIR,
                                         target_size=(ROWS, COLS),
                                         classes=FISH_CLASSES,
                                         batch_size=batch_size*2,
                                         shuffle=False)

  model.fit_generator(train_flow,
                      samples_per_epoch=n_train_samples,
                      nb_epoch=nb_epoch,
                      validation_data=val_flow,
                      nb_val_samples=n_val_samples,
                      verbose=1,
                      callbacks=[checkpointer, early_stopping, tensorboard])


def test():
  # ensure correct ordering
  dir_files = [im for im in os.listdir(TEST_DIR)]
  file_ext = os.path.splitext(dir_files[0])[1]
  filenames_orig = [os.path.basename(x).split('-')[0] for x in dir_files]
  filenames_orig = sorted(set(filenames_orig))

  filenames = []
  for filename_orig in filenames_orig:
    filenames.extend([filename_orig + '-' + str(i) + file_ext for i in range(N_PROPOSALS)])

  predictions_filename = TEMP_DIR + '/predictions' + BEST_WEIGHTS + '.npy'
  if os.path.isfile(predictions_filename):
    predictions = np.load(predictions_filename)
  else:
    # load model
    model = create_model()
    model.load_weights('/dev/shm/' + MODEL_NAME + '/models/weights.' + BEST_WEIGHTS + '.hdf5')
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='categorical_crossentropy')

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    batch_size = 256
    dataflow = datagen.flow_from_directory(TEST_DIR,
                                           filenames=filenames,
                                           target_size=(ROWS, COLS),
                                           class_mode=None,
                                           batch_size=batch_size,
                                           shuffle=False)

    n_files = len(filenames)
    predictions = model.predict_generator(dataflow, n_files)
    np.save(predictions_filename,predictions)

  print(predictions.shape)

  # rearange prediction in array
  predictions = predictions.reshape((len(filenames_orig), N_PROPOSALS, len(FISH_CLASSES)))

  no_fish_ind = FISH_CLASSES.index('NoF')
  # select the maximum scores along all crops for each fish
  final_predicitons = np.max(predictions, axis=1)
  # select the minimum for along all crops for no fish
  no_fish_max_scores = np.min(predictions[:, :, no_fish_ind], axis=1)
  final_predicitons[:, no_fish_ind] = no_fish_max_scores
  # normalize probs
  final_predicitons = final_predicitons / np.sum(final_predicitons, axis=1)[:, None]

  print(final_predicitons.shape)

  final_predicitons_classes = np.argmax(final_predicitons, axis=1)
  final_predicitons_txt = [FISH_CLASSES[i] for i in final_predicitons_classes]

  filenames_orig_ext = [x + '.jpg' for x in filenames_orig]

  txt = '\n'.join([f + ' ' + c for f, c in zip(filenames_orig_ext, final_predicitons_txt)])
  with open('predictions.txt', 'w') as f:
    f.write(txt)

  final_predicitons_one_hot = np.zeros(final_predicitons.shape, dtype=np.float32)
  final_predicitons_one_hot[(np.arange(final_predicitons.shape[0]),
                                       final_predicitons_classes)] = 1.0

  final_predicitons_classes_ = np.argmax(final_predicitons_one_hot, axis=1)
  assert(np.array_equal(final_predicitons_classes_, final_predicitons_classes))

  final_predicitons = (final_predicitons + final_predicitons_one_hot) * 0.5


  submission = pd.DataFrame(final_predicitons_one_hot, columns=FISH_CLASSES)
  submission.insert(0, 'image', filenames_orig_ext)

  print(submission.head())
  now = datetime.datetime.now()
  sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '_one_hot.csv'
  submission.to_csv(sub_file, index=False)


  submission = pd.DataFrame(final_predicitons, columns=FISH_CLASSES)
  submission.insert(0, 'image', filenames_orig_ext)

  print(submission.head())
  now = datetime.datetime.now()
  sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
  submission.to_csv(sub_file, index=False)


def eval_img(image_name):
  file_ext = '.jpg'
  filenames = [image_name + '-' + str(i) + file_ext for i in range(N_PROPOSALS)]

  # load model
  model = create_model()
  model.load_weights('/dev/shm/' + MODEL_NAME + '/models/weights.' + BEST_WEIGHTS + '.hdf5')
  model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=1e-4), loss='categorical_crossentropy')

  datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  batch_size = 256
  dataflow = datagen.flow_from_directory(TEST_DIR,
                                         filenames=filenames,
                                         target_size=(ROWS, COLS),
                                         class_mode=None,
                                         batch_size=batch_size,
                                         shuffle=False)

  n_files = len(filenames)
  predictions = model.predict_generator(dataflow, n_files)

  # rearange prediction in array
  predictions = predictions.reshape((1, N_PROPOSALS, len(FISH_CLASSES)))

  no_fish_ind = FISH_CLASSES.index('NoF')
  # select the maximum scores along all crops for each fish
  final_predicitons = np.max(predictions, axis=1)
  # select the minimum for along all crops for no fish
  no_fish_max_scores = np.min(predictions[:, :, no_fish_ind], axis=1)
  final_predicitons[:, no_fish_ind] = no_fish_max_scores
  # normalize probs
  final_predicitons = final_predicitons / np.sum(final_predicitons, axis=1)[:, None]

  final_predicitons_classes = np.argmax(final_predicitons, axis=1)
  final_predicitons_txt = [FISH_CLASSES[i] for i in final_predicitons_classes]

  print(final_predicitons_txt)
  print(np.round(final_predicitons, decimals=2))

if __name__ == '__main__':
  # train()
  test()
  # eval_img('img_00119')
