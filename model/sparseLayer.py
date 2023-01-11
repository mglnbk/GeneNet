import tensorflow as tf
import itertools
import sys
sys.path.append('/home/sunzehui/GeneNet')
from data_utils.pathways.reactome import Reactome, ReactomeNetwork
import pandas as pd
import numpy as np
from tensorflow.keras import layers


def get_map_from_layer(layer_dict):
  """
  :param: layer_dict: dictionary of connections (e.g {'pathway1': ['g1', 'g2', 'g3']}
  :return: dataframe map of layer (index = genes, columns = pathways, values = 1 if connected; 0 else)
  """
  pathways = layer_dict.keys()
  genes = list(itertools.chain.from_iterable(layer_dict.values()))
  genes = list(np.unique(genes))
  df = pd.DataFrame(index=pathways, columns=genes)
  for k, v in layer_dict.items():
    df.loc[k, v] = 1
  df = df.fillna(0)
  return df.T


class SparseLinear(layers.Layer):
  def __init__(self, _mapp):
    super().__init__()
    if type(_mapp)==pd.DataFrame:
      self.mapp = _mapp.values
    else:
      self.mapp = _mapp
    
  def build(self, input_shape):
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.mapp.shape[1])),
        trainable=True,
        dtype='float32'
        )
    self.mask = tf.Variable(initial_value = self.mapp, trainable = False, dtype = 'float32')
    
  def call(self, input):
    return tf.matmul(input, self.mask * self.w)

  def getWeight(self):
    return self.w


net = ReactomeNetwork()
layers = net.get_layers(n_levels=2)

model = tf.keras.Sequential()

for i, layer in enumerate(layers[::-1]):
  mapp = get_map_from_layer(layer)
  mask = mapp.to_numpy()
  print(mask.shape)
  model.add(SparseLinear(mask))

model(tf.ones((10, 11336)))
model.summary()