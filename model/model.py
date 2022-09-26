import tensorflow as tf
from tensorflow import keras
from keras import layers

vocab_size = 20000  # Only consider the top 20k words
num_tokens_per_example = 200  # Only consider the first 200 words of each movie review
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=num_tokens_per_example
)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=num_tokens_per_example)

