import tensorflow as tf
from config_path import *
from model.data import Dataset

# Dataset Loading
ds = Dataset(training=True, data_type='cnv')

# 