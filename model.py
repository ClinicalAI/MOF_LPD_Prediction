epoch = 200
batch = 64
prefix="MOF"
save_dir = f"./models/{prefix}_{epoch}/"



import os
import subprocess
import math
import numpy as np
import random
from scipy import ndimage
from sklearn import preprocessing
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
import tensorflow.keras as keras
from keras import backend as K
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.framework.ops import Tensor
from numba import cuda
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)

try:
    subprocess.run(['mkdir','-p', save_dir],check=True)
except Exception as e:
    raise


def load_data():

  data = np.load('./data/train_data.npz',allow_pickle=True)

  x = data["X"]
  print(x.shape)
  y = data["Y"]
  print(y.shape)

  return x, y

def load_data_unseen():

   data = np.load('./data/unseen_data.npz',allow_pickle=True)

   x = data["X"]
   print(x.shape)
   y = data["Y"]
   print(y.shape)

   return x, y

@tf.function
def rotate(volume):
    def scipy_rotate(volume):
        angles = [-5,-10,-15,0,5,10,15]
        axes = [(0,1),(0,2),(1,2)]
        volume = ndimage.rotate(volume,
                                angle=random.choice(angles),
                                axes=random.choice(axes),
                                mode = 'nearest',
                                reshape=False)
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    volume = rotate(volume)
    return volume, label

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_model(width=50, height=50, depth=50, channel=4):

    inputs = layers.Input((width, height, depth, channel))

    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(inputs)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu")(x)
    x = layers.AveragePooling3D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)


    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.AveragePooling3D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)


    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.AveragePooling3D(pool_size=2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=512, activation="relu",kernel_constraint=tf.keras.constraints.max_norm(2.))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(units=256, activation="relu",kernel_constraint=tf.keras.constraints.max_norm(2.))(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs, outputs)
    return model


model = get_model(50, 50, 50,4)
model.summary()
plot_model(model, to_file=save_dir+'model_struct.png', show_shapes=True)

"""# Models Training"""

def train_model(epoch,_batch,prefix,channel=False):

  _epoch = epoch
  n_splits=5
  kf = KFold(n_splits=n_splits,shuffle=False)

  save_dir = f"models/{prefix}_{epoch}/"

  try:
      subprocess.run(['mkdir','-p', save_dir],check=True)
  except Exception as e:
      raise

  i = 1

  x, y = load_data()
  # deleting unnecessary channels
  x = np.delete(x,5,4)
  x = np.delete(x,4,4)
  x = np.delete(x,3,4)
  print(x.shape)
  y_pred_list = []
  y_test_list = []
  r2_values=[]
  r2_values_val=[]
  epochs_list = []
  folds = []

  for train_index, test_index in kf.split(x,y):

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = (train_loader
                     .repeat(5)
                     .map(train_preprocessing,num_parallel_calls=4)
                     .shuffle(5*x_train.shape[0])
                     .cache()
                     .batch(_batch, drop_remainder=True)
                     .prefetch(2))

    validation_dataset = (
      validation_loader
      .batch(_batch, drop_remainder=True)
      )

    model = get_model(width=x_train.shape[1], height=x_train.shape[2], depth=x_train.shape[3],channel=x_train.shape[4])


    METRICS = [
                tf.keras.metrics.MeanSquaredError(name='mean_square_error'),
                tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error"),
                r2_keras,
                ]

    model.compile(
                loss='mse', optimizer='adam',
                metrics=METRICS)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+ prefix + 'model_1_epoch_{}_{}.h5'.format(_epoch,i),
                                                    monitor='val_r2_keras', verbose=1,
                                                    save_best_only=True, mode='max')
    log = tf.keras.callbacks.CSVLogger(save_dir+ prefix + 'model_1_epoch_{}_{}.log'.format(_epoch,i))


    print('\n')
    print('\n')
    print('**************************************************')
    print('training on fold {} out of {}'.format(i,n_splits))
    print('**************************************************')
    print('\n')
    print('\n')

    history = model.fit(
        train_dataset,
        epochs=_epoch,
        validation_data=validation_dataset,
        callbacks=[ checkpoint,log],
        verbose=2
    )

    model.load_weights(save_dir+ prefix + 'model_1_epoch_{}_{}.h5'.format(_epoch,i))

    y_pred = model.predict(x_test)


    y_pred_list.append(y_pred.flatten())
    y_test_list.append(y_test.flatten())
    r2_value = r2_score(y_test, y_pred)
    r2_values.append(history.history['r2_keras'])
    r2_values_val.append(history.history['val_r2_keras'])

    df = pd.DataFrame({'y_true': y_test.flatten(), 'y_predicted': y_pred.flatten()})

    epoch_list = [ j+1 for j in range(_epoch)]
    fold_list = [ i for j in range(_epoch)]

    epochs_list.append(epoch_list)
    folds.append(fold_list)

    p = np.polyfit(y_test,y_pred,1)


    _x = [np.amin(y_test), np.amax(y_test)]

    _y = [p[0][0]*_x[0] + p[1][0], p[0][0]*_x[1] + p[1][0]]


    plt.figure()
    sns.regplot(x="y_true", y="y_predicted", data=df)
    plt.text(_x[1]-2.5,_y[1]+2.5, f'$R^2={r2_value:.2f}$')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([_x[0]-5, _x[1]+5])
    plt.ylim([_x[0]-5, _x[1]+5])
    plt.title(f"Fold {i}")
    plt.tight_layout()
    plt.savefig(save_dir + prefix +f'r2_model_1_epoch_{_epoch}_fold{i}.png', dpi=600)

    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.title(f"Fold {i}")
    plt.tight_layout()
    plt.savefig(save_dir + prefix +f'model_1_epoch_{_epoch}_fold{i}.png', dpi=600)

    plt.figure()
    plt.plot(history.history['val_r2_keras'], label='R2')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.title(f"Fold {i}")
    plt.tight_layout()
    plt.savefig(save_dir + prefix +f'r2_hist_model_1_epoch_{_epoch}_fold{i}.png', dpi=600)


    plt.figure()
    plt.plot(history.history['mean_square_error'], label='Mean Square Error')
    plt.plot(history.history['mean_absolute_error'], label='Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.title(f"Fold {i}")
    plt.tight_layout()
    plt.savefig(save_dir + prefix +f'mse_model_1_epoch_{_epoch}_fold{i}.png', dpi=600)


    i += 1

  fold_df = pd.DataFrame({'fold_number': (np.array(folds)).flatten(),
                           'epoch': (np.array(epochs_list)).flatten(),
                           'r2': (np.array(r2_values)).flatten(),
                           'r2_val':(np.array(r2_values_val)).flatten()})

  y_pred_list = [item for sublist in y_pred_list for item in sublist]
  print(y_pred_list[:10])

  y_test_list = [item for sublist in y_test_list for item in sublist]
  print(y_test_list[:10])
  true_test_df = pd.DataFrame({'y_pred': y_pred_list,
                           'y_test': y_test_list})

  fold_df.to_csv(save_dir + prefix +'fold_df.csv')
  true_test_df.to_csv(save_dir + prefix +'_true_test_df.csv')

  r2_ = r2_score(true_test_df['y_test'], true_test_df['y_pred'])
  plt.figure()
  sns.regplot(x="y_test", y="y_pred", data=true_test_df)
  plt.text(_x[1]-2.5,_y[1]+2.5, f'$R^2={r2_:.2f}$')
  plt.xlabel('True value')
  plt.ylabel('Predicted value')
  plt.xlim([_x[0]-5, _x[1]+5])
  plt.ylim([_x[0]-5, _x[1]+5])
  plt.title(f"{prefix}")
  plt.tight_layout()
  plt.savefig(save_dir + prefix +f'r2_model_1_epoch_{_epoch}.png', dpi=600)

  plt.figure()
  sns.lineplot(x="epoch", y="r2", data=fold_df)
  sns.lineplot(x="epoch", y="r2_val", data=fold_df)
  r2 = Line2D([0], [0], color='blue')
  r2_val = Line2D([0], [0], color='orange')
  plt.xlabel('Epoch')
  plt.ylabel('R2')
  plt.legend([r2, r2_val],['R2', 'R2 validation'])
  plt.title(f"{prefix}")
  plt.tight_layout()
  plt.savefig(save_dir + prefix +f'r2hist_model_1_epoch_{_epoch}.png', dpi=600)

#evaluation for unseen data
def eval_model(epoch,_batch,prefix,channel=False):

  _epoch = epoch
  n_splits=5
  kf = KFold(n_splits=n_splits,shuffle=False)

  save_dir = f"models/{prefix}_{epoch}/"

  x, y = load_data_unseen()
  x = np.delete(x,5,4)
  x = np.delete(x,4,4)
  x = np.delete(x,3,4)

  print(x.shape)
  y_pred_list = []
  y_test_list = []
  r2_values=[]
  r2_values_val=[]
  epochs_list = []
  folds = []

  for i in range(n_splits):

    model = get_model(width=x.shape[1], height=x.shape[2], depth=x.shape[3],channel=x.shape[4])

    for layer in model.layers:
        layer.trainable = False

    model.compile(
                loss='mse', optimizer='adam'
                )

    print('\n')
    print('\n')
    print('**************************************************')
    print('training on fold {} out of {}'.format(i,n_splits))
    print('**************************************************')
    print('\n')
    print('\n')

    model.load_weights(save_dir+ prefix + 'model_1_epoch_{}_{}.h5'.format(_epoch,i+1))

    y_pred = model.predict(x)


    y_pred_list.append(y_pred.flatten())
    y_test_list.append(y.flatten())
    r2_value = r2_score(y, y_pred)

    df = pd.DataFrame({'y_true': y.flatten(), 'y_predicted': y_pred.flatten()})

    epoch_list = [ j+1 for j in range(_epoch)]
    fold_list = [ i for j in range(_epoch)]

    epochs_list.append(epoch_list)
    folds.append(fold_list)

    p = np.polyfit(y,y_pred,1)
    _x = [np.amin(y), np.amax(y)]
    _y = [p[0][0]*_x[0] + p[1][0], p[0][0]*_x[1] + p[1][0]]


    plt.figure()
    sns.regplot(x="y_true", y="y_predicted", data=df)
    plt.text(_x[1]-2.5,_y[1]+2.5, f'$R^2={r2_value:.2f}$')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    plt.xlim([_x[0]-5, _x[1]+5])
    plt.ylim([_x[0]-5, _x[1]+5])
    plt.title(f"Unseen data fold:{i+1}")
    plt.tight_layout()
    plt.savefig(save_dir + prefix +f'r2_epoch_{_epoch}_fold{i+1}_unseen.png', dpi=600)


train_model(epoch,batch,prefix=prefix)
eval_model(epoch,batch,prefix=prefix)
