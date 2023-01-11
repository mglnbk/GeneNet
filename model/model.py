import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import backend
from keras import Sequential
from encoder import encoder
from decoder import sparse_decoder

class gene_net(tf.keras.Model):
    def __init__(self, ):
        super().__init__()
        self.encoder = encoder
        self.decoder = sparse_decoder
        
    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask) 
