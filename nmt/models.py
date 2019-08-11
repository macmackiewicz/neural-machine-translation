import numpy as np

from keras.models import Model
from keras.layers import (
    Embedding, Dense, LSTM, Bidirectional, Dropout, Input, Concatenate
)


class Sequence2Sequence:
    @classmethod
    def of(cls, source_vocab_size: int, target_vocab_size: int,
           target_max_sentence_length: int, n_units: int=256,
           dropout: float=0.1) -> 'Sequence2Sequence':

        encoder_inputs = Input(shape=(None,))
        encoder_embeddings = Embedding(source_vocab_size + 1, n_units,
                                       mask_zero=True)(encoder_inputs)
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = \
            Bidirectional(
                LSTM(n_units, return_state=True, return_sequences=True)
            )(encoder_embeddings)

        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embeddings = Embedding(target_vocab_size + 1, n_units,
                                       mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(n_units * 2, return_sequences=True,
                            return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embeddings,
                                             initial_state=encoder_states)
        decoder_dense = Dense(target_vocab_size, activation='softmax')
        dense_output = decoder_dense(decoder_outputs)
        dropout_output = Dropout(dropout)(dense_output)

        training_model = Model([encoder_inputs, decoder_inputs], dropout_output)

        training_model.compile(loss='categorical_crossentropy',
                               optimizer='adam', metrics=['accuracy'])

        # inference model
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(n_units * 2,))
        decoder_state_input_c = Input(shape=(n_units * 2,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embeddings, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        return cls(target_max_sentence_length, training_model, encoder_model,
                   decoder_model)

    def __init__(self, target_max_sentence_length: int, training_model: Model,
                 encoder_model: Model, decoder_model: Model):
        self.target_max_sentence_length = target_max_sentence_length
        self.training_model = training_model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def fit(self, *args, **kwargs):
        return self.training_model.fit(*args, **kwargs)

    def predict_sequence(self, sequence: np.ndarray) -> list:
        state = self.encoder_model.predict(sequence)

        predicted_sequence = []
        # initialise target sequence with start of sequence token index
        target_sequence = np.array([0])

        while len(predicted_sequence) <= self.target_max_sentence_length:
            predicted_tokens, h, c = self.decoder_model.predict(
                [target_sequence] + state
            )
            # TODO: replace with BEAM search
            predicted_word_index = np.argmax(predicted_tokens[0, 0, :])
            if predicted_word_index == 0:
                break
            predicted_sequence.append(predicted_word_index)

            target_sequence = np.array([predicted_word_index])
            state = [h, c]

        return predicted_sequence
