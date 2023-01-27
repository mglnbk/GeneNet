import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


class encoder(layers.Layer):
    """encoder layer(Dense Layer, encode genetic profile data into
    condensed latent layer, stakced)

    Args:
        layers (_type_): _description_
    """
    def __init__(self, units, input_dim, regularizer, initializer, name=None):
        super().__init__()
        self.initializer = initializer
        self.regularizer = regularizer
        self.w = self.add_weight(shape = (input_dim, units),
                                 initializer = self.initializer,
                                 regularizer = self.regularizer,
                                 trainable= True
                                 )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True) 
    
    def build(self, input_shape):
        return super().build(input_shape)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

