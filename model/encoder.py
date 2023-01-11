import numpy as np
import pandas as pd
import tensorflow as tf


class encoder(tf.keras.layers.Layer):
    def __init__(self, _data):
        super().__init__()
        self.data = _data
    
    def build(self, input_shape):
        return super().build(input_shape)
    
    def call(self, inputs, training, mask=None):
        return super().call(inputs, training, mask)