import torch
import itertools
from GeneNet_data.pathways.reactome import Reactome, ReactomeNetwork
import pandas as pd
import numpy as np
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


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


class ZeroFilter(nn.Module):
  def __init__(self, _mapp, _mask):
    super().__init__()
    self.mapp = _mapp  # dataFrame
    self.fc = nn.Linear(in_features=_mapp.shape[0], out_features=_mapp.shape[1])
    with torch.no_grad():
      self.fc.weight.mul_(_mask.T)

  def _get_name(self):
    return "Filter"

  def forward(self, _x):
    return F.relu(self.fc(_x))

  def getWeight(self):
    return self.weight


class SparseLinear(nn.Module):
  def __init__(self, in_features, out_features, sparse_indices):
    super(SparseLinear, self).__init__()
    self.weight = nn.Parameter(data=torch.sparse.FloatTensor(sparse_indices, torch.randn(sparse_indices.shape[1]),
                                                             [in_features, out_features]), requires_grad=True)
    self.bias = nn.Parameter(data=torch.randn(out_features), requires_grad=True)

  def forward(self, _x):
    return torch.sparse.addmm(self.bias, self.weight, _x, 1., 1.)


net = ReactomeNetwork()
layers = net.get_layers(n_levels=2)

model = nn.Sequential()

for i, layer in enumerate(layers[::-1]):
  mapp = get_map_from_layer(layer)
  # if i == 0:
  #   genes = list(mapp.index)[0:10]
  # filter_df = pd.DataFrame(index=genes)
  # _all = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
  mask = torch.tensor(mapp.to_numpy(), requires_grad=False)
  model.add_module(str(i), module=ZeroFilter(mapp, _mask=mask))

x = torch.randn(size=(10, 11336))
summary(model, input_size=(10, 1, 11336))
