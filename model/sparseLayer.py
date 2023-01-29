import tensorflow as tf
import itertools
import sys  
sys.path.append('/home/sunzehui/GeneNet')
from data_utils.pathways.reactome import Reactome, ReactomeNetwork
import pandas as pd
import numpy as np
from tensorflow.keras import layers
import logging
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform
from tensorflow.keras.regularizers import L1, L2
from tensorflow.keras.layers import Dropout, ReLU
from tensorflow.keras.activations import tanh, sigmoid, relu

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

def get_layer_maps(genes, n_levels, direction='root_to_leaf', add_unk_genes=True):
    """Obtain layer mappings 

    Args:
        genes (list): selected_genes for latent layers
        n_levels (int): number of decode layers
        direction (str): default to 'root_to_leaf'
        add_unk_genes (bool): default to True

    Returns:
        list: [df1, df2, df3, df4, ...]
    """
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes # gene
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print('filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        print('filtered_map', filter_df.shape)

        if add_unk_genes:
            print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1

        filtered_map = filtered_map.fillna(0)
        print('filtered_map', filter_df.shape)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps

class SparseLinear(layers.Layer):
  def __init__(
    self, 
    _mapp, 
    constraint=NonNeg, 
    regularizer="L2",
    initializer="GlorotUniform",
    use_bias=True,
    dropout_rate=.0
  ):
    """Creat tensorflow kernel

    Args:
        _mapp (pd.DataFrame): _description_
        constraint (_type_, optional): _description_. Defaults to NonNeg.
        regularizer (str, optional): _description_. Defaults to "L2".
        initializer (str, optional): _description_. Defaults to "GlorotUniform".
        use_bias (bool, optional): _description_. Defaults to True.
        dropout_rate (float, optional): _description_. Defaults to 0.
    """
    super().__init__()
    if type(_mapp)==pd.DataFrame:
      self.mapp=_mapp.values
    else:
      self.mapp=_mapp
    if regularizer == "L2":
      self.regularizer=L2
    else:
      self.regularizer=L1
    if initializer == "GlorotUniform":
      self.initializer=GlorotUniform
    else:
      self.initializer=GlorotNormal
    self.constraint=constraint
    self.use_bias = use_bias
    self.bias = None
    self.units = _mapp.shape[-1]
    self.dropout_rate = dropout_rate
    self.dropout=None
    
  def build(self, input_shape):
    last_dim = tf.compat.dimension_value(input_shape[-1])
    self.kernel=self.add_weight(
      "kernel",
      shape=[last_dim, self.units],
      initializer=self.initializer,
      regularizer=self.regularizer,
      constraint=self.constraint,
      trainable=True,
      dtype=self.dtype
      )
    self.mask = tf.Variable(
      initial_value = self.mapp, 
      trainable = False, 
      dtype = self.dtype
      )
    if self.use_bias == True:
      self.bias = self.add_weight(
        "bias",
        shape = [self.units,],
        initializer=self.initializer,
        regularizer=self.regularizer,
        constraint=self.constraint,
        trainable=True,
        dtype=self.dtype
      )
    
  def call(self, input):
    if self.use_bias:
      outputs=tf.matmul(input, self.mask * self.kernel)+self.bias
    else:
      outputs=tf.matmul(input, self.mask * self.kernel)
    if self.dropout_rate!=0:
      self.dropout=Dropout(rate=self.dropout_rate,
                           noise_shape=self.units)
      outputs=self.dropout(outputs)
    
    return outputs

  def getWeight(self):
    return self.kernel

class decoder(layers.Layer):
  def __init__(
    self, 
    genes,
    nb_layers, 
    # activation, 
    # dropout, 
    # initializer, 
    # regularizer,
    # constraint
    ):
    """Decoder Layers biologically informed

    Args:
        genes (list): Latent layer gene, shuffled
        nb_layers (int): number of layers
        activation (str): 'relu' or 'tanh'
        dropout (float): if 0, then no dropout
        initializer (str): "GolorotUniform" or "GolorotNormal"
        regularizer (list): l2
        constraint (str): 'NonNeg'
    """
    super().__init__()
    self.genes=genes
    self.nb_layers=nb_layers
    # self.activation=activation
    # self.initializer=initializer
    # self.regularizer=regularizer
    # self.constraint=constraint
    
    self.decoder_list=[]
    self.activation_list=[]

  def build(self, input_shape):
    mapp_dict=get_layer_maps(self.genes, n_levels=self.nb_layers)

    for nb, i in enumerate(mapp_dict[:-1]):
      self.decoder_list.append(
        SparseLinear(_mapp=i, 
                     initializer="GolorotUniform", 
                     regularizer="L2", 
                     use_bias=True,
                     dropout_rate=.3)
                               )
      print(i.shape)
    

  def call(self, inputs):
    outputs=inputs
    for decoder_layer in self.decoder_list:
      outputs = decoder_layer(outputs)
      outputs = relu(outputs)
    
    return outputs
      

if __name__ == "__main__":
  net = ReactomeNetwork()
  layers = net.get_layers(n_levels=3)
  model = tf.keras.Sequential()
  genes = ['ABC1', 'TP53', 'PTEN']
  
  d = decoder(genes=genes, nb_layers=4)
  print(d(tf.ones((10, 3))))