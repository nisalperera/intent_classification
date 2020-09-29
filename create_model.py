import tensorflow as tf
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.metrics import Precision, Recall
from keras.optimizers import Adam


def createmodel(vocab_size, embedding_size, seq_len, num_of_classes, examples):

    # print(num_of_classes)
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size))
    # model.add(Bidirectional(LSTM(seq_len * 6, input_shape=(seq_len, embedding_size), return_sequences=True)))
    # model.add(Bidirectional(LSTM(seq_len * 5, input_shape=(seq_len, embedding_size), return_sequences=True)))
    model.add(Bidirectional(LSTM(seq_len * 4, return_sequences=True)))
    model.add(Bidirectional(LSTM(seq_len * 3, return_sequences=True)))
    model.add(Bidirectional(LSTM(seq_len * 2, return_sequences=True)))
    model.add(Bidirectional(LSTM(seq_len)))
    model.add(Dense(num_of_classes, activation="softmax"))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])

    model.summary()
    return model

# createmodel(1000, 100, 100)
