import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (
    Embedding, Dense, LSTM, TimeDistributed, RepeatVector, Bidirectional,
    Dropout
)


class Sequence2Sequence:
    def __init__(self, source_vocab_size, target_vocab_size, #pylint: disable=R0913
                 source_max_sentence_length, target_max_sentence_length,
                 n_units=256):
        model = Sequential()
        model.add(Embedding(source_vocab_size,
                            n_units,
                            input_length=source_max_sentence_length,
                            mask_zero=True))
        model.add(Bidirectional(LSTM(n_units, activation='relu')))
        model.add(Dropout(0.2))
        model.add(RepeatVector(target_max_sentence_length))
        model.add(LSTM(n_units, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(TimeDistributed(
            Dense(target_vocab_size, activation='softmax')))

        optimizer = Adam(lr=1e-3, decay=1e-5)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        self._model = model

    def __getattr__(self, item):
        return getattr(self._model, item)

    def predict_sequence(self, sequence: np.ndarray) -> list:
        prediction = self.predict(sequence, verbose=0)[0]

        return [np.argmax(vector) for vector in prediction]
