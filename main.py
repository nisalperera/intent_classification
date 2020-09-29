from math import trunc

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from numpy import argmax
from tensorflow.keras.models import load_model

from create_model import createmodel
from data_preprocessing import Preprocessor


def model_utils(preprocessor):
    vocab_size, max_len, training_padded, training_intent_list = preprocessor.load_and_preprocess_training_data()

    testing_padded, testing_intent_list = preprocessor.load_and_preprocess_eval_data(training_intent_list.shape[1])

    # if batchsize // 8 != 0:
    #     batchsize = trunc(batchsize / 8) * 8

    # model = createmodel(vocab_size, 100, max_len, training_intent_list.shape[1], training_padded.shape[0])
    #
    # print(training_intent_list[:10])
    # print(training_padded[:10])
    #
    # tensorboard = TensorBoard(log_dir='.\\logs')
    # checkpoint = ModelCheckpoint("./model/model.ckpt", monitor='val_loss', verbose=1, save_best_only=True)
    # earlystop = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    #
    # model.fit(training_padded, training_intent_list, verbose=1, validation_split=0.1, epochs=100, batch_size=64,
    #           callbacks=[tensorboard, checkpoint, earlystop])
    #
    # results = model.evaluate(testing_padded, testing_intent_list)
    # print(model.metrics_names)
    # print(results)


def inference(text, preprocessor):
    tokenized_text = preprocessor.preprocesses_sequence(text)
    model = load_model("./model/model.ckpt")
    intent = list(preprocessor.intents.keys())[list(preprocessor.intents.values()).index(model.predict_classes(tokenized_text))]
    print(intent)


if __name__ == "__main__":
    preprocessor = Preprocessor()
    model_utils(preprocessor)
    inference("what kind of ground transportation is available in denver", preprocessor)