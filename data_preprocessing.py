from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from pandas import read_csv

from create_model import createmodel


class Preprocessor:

    def __init__(self):
        self.intents = {}
        self.tokenizer = Tokenizer()
        self.max_len = 0
        self.training_intent_list = []

    def load_and_preprocess_training_data(self):
        # with open("./data/atis_intents_train.csv") as f:
        train_data = read_csv("./data/atis_intents_train.csv")
        # train_data.info()

        x_train = train_data.iloc[:, 1].values
        y_train = train_data.iloc[:, 0].values

        batch_size = train_data.iloc[:, 0].groupby(train_data.iloc[:, 0]).count()
        # print(y_train.reshape(y_train.shape[0], 1).shape)
        # print(x_train.shape)

        intent_numbers = []
        for intent in y_train:
            if intent not in self.intents:
                self.intents[intent] = len(self.intents)
        for key, value in self.intents.items():
            intent_numbers.append(len(train_data.loc[train_data.iloc[:, 0] == key]))
            print(len(train_data.loc[train_data.iloc[:, 0] == key]))

        ytrain = []
        for intent in y_train:
            ytrain.append(self.intents[intent])

        vocab_size = self.tokenize_sequences(x_train)
        # print(tokenizer.word_index)
        # print(vocab_size)
        # for sentence in x_train:
        tokenized = self.tokenizer.texts_to_sequences(x_train)
        self.max_len = self.max_length(x_train)
        training_padded = pad_sequences(tokenized, maxlen=self.max_len, padding="post", truncating="post")
        training_intent_list = to_categorical(ytrain, num_classes=len(self.intents.values()))
        # print(training_padded.shape)
        # print(max_len)
        # print(training_intent_list.shape)
        # print("Vacab Size: ", vocab_size)

        return vocab_size, self.max_len, training_padded, training_intent_list

    def load_and_preprocess_eval_data(self, num_classes):
        train_data = read_csv("./data/atis_intents_test.csv")

        x_test = train_data.iloc[:, 1].values
        y_test = train_data.iloc[:, 0].values
        # print(y_train.reshape(y_train.shape[0], 1).shape)
        # print(x_train.shape)

        ytrain = []
        for intent in y_test:
            ytrain.append(self.intents[intent])

        vocab_size = self.tokenize_sequences(x_test)

        tokenized = self.tokenizer.texts_to_sequences(x_test)
        testing_padded = pad_sequences(tokenized, maxlen=self.max_len, padding="post", truncating="post")
        testing_intent_list = to_categorical(ytrain, num_classes=num_classes)

        return testing_padded, testing_intent_list

    def preprocesses_sequence(self, sequence):
        text = self.tokenizer.texts_to_sequences([sequence])
        text_padded = pad_sequences(text, maxlen=self.max_len, padding="post", truncating="post")

        return text_padded

    def tokenize_sequences(self, sequences):
        self.tokenizer.fit_on_texts(sequences)
        vocab_size = len(self.tokenizer.word_index) + 1

        return vocab_size

    def tokenize_labels(self, labels):
        # for sentence in sequences:
        self.tokenizer.fit_on_texts(labels)

        return self.tokenizer

    def max_length(self, sequences):
        return max([len(d.split()) for d in sequences])
