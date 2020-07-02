import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM


def createmodel(vocab_size, embedding_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=seq_len, mask_zero=True))
    model.add(LSTM(32))

    model.compile(optimizer="adam", loss="categorical_crossentropy")

    model.summary()


createmodel(1000, 100, 100)
