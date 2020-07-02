from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from pandas import read_csv


def load_data():
    # with open("./data/atis_intents_train.csv") as f:
    train_data = read_csv("./data/atis_intents_train.csv")
    train_data.info()

    x_train = train_data.iloc[:, 1].values
    y_train = train_data.iloc[:, 0].values
    print(y_train.shape)
    # print(x_train[:5])

    tokenizer, vocab_size = tokenize_sequence(x_train)
    # print(tokenizer.word_index)
    # print(vocab_size)
    # for sentence in x_train:
    tokenized = tokenizer.texts_to_sequences(x_train)
    max_len = max_length(x_train)
    padded = pad_sequences(tokenized, maxlen=max_len)
    categories = to_categorical(len(y_train), num_classes=y_train.shape)
    # print(padded.shape)
    # print(max_len)
    print(categories)


def tokenize_sequence(sequences):
    tokenizer = Tokenizer()
    # for sentence in sequences:
    tokenizer.fit_on_texts(sequences)
    vocab_size =len(tokenizer.word_index) + 1

    return tokenizer, vocab_size


def max_length(sequences):
    return max(len(d.split()) for d in sequences)

load_data()