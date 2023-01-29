import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import Sequential
from encoder import encoder
from sparseLayer import decoder

class gene_net(tf.keras.Model):
    def __init__(self, ):
        super().__init__()
        self.encoder = encoder()
        self.decoder = decoder(genes=['TP53', 'PTEN', 'ABC1', 'MYC'],
                               nb_layers=4)
        
    def call(self, inputs, training=None, mask=None):
        return self.decoder(self.encoder(inputs))



if __name__ == "__main__":
    gnet = gene_net()
    gnet(tf.ones(shape=(1011, 10)))
    gnet.summary()
