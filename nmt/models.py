from typing import Tuple, List

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
            Bidirectional(LSTM(n_units, return_state=True))(encoder_embeddings)

        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embeddings = Embedding(target_vocab_size + 1, n_units,
                                       mask_zero=True)(decoder_inputs)
        # returned states are not used for training model,
        # but will be used for inference
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
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

        return cls(target_max_sentence_length, training_model, encoder_model,
                   decoder_model)

    def __init__(self, target_max_sentence_length: int, training_model: Model,
                 encoder_model: Model, decoder_model: Model):
        self.target_max_sentence_length = target_max_sentence_length
        self.training_model = training_model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def _decode_next_token(self, token: np.array, state: np.array) \
            -> Tuple[np.array, np.array, np.array]:
        next_token_predictions, h, c = self.decoder_model.predict(
            [token] + state
        )
        return next_token_predictions[0, 0, :], h, c

    def predict_sequence(self, sequence: np.ndarray, beam_width=3) -> List[int]:
        state = self.encoder_model.predict(sequence)

        # decode initial tokens using start-of-sequence token
        next_token_predictions, h, c = self._decode_next_token(np.array([0]),
                                                               state)

        # take tokens with highest probabilities up to beam_width
        candidate_tokens = next_token_predictions.argsort()[-beam_width:][::-1]
        state = [h, c]

        # tuple with sequence probability, decoded sequence
        # and decoded sequence's state
        sequence_candidates = [
            (np.log(next_token_predictions[idx]), [idx], state)
            for idx in candidate_tokens
        ]

        for _ in range(self.target_max_sentence_length):
            new_sequence_candidates = []

            for probability, target_sequence, state in sequence_candidates:
                # sequences that reached end-of-sequence tokens
                # are kept as candidates
                if target_sequence[-1] == 0:
                    new_sequence_candidates.append((probability,
                                                    target_sequence, state))
                    continue

                next_token_predictions, h, c = self._decode_next_token(
                    np.array([target_sequence[-1]]), state)

                # logarithm of tokens' probabilities
                # given probability of a sequence decoded so far
                sequence_conditional_probability = \
                    np.log(next_token_predictions) + probability

                for idx, p in enumerate(sequence_conditional_probability):
                    new_sequence_candidates.append(
                        (p, target_sequence + [idx], [h, c])
                    )
            sequence_candidates = sorted(new_sequence_candidates,
                                         key=lambda x: x[0])[-beam_width:]

        # return sequence with highest logarithm of conditional probabilities
        return sorted(sequence_candidates, key=lambda x: x[0])[-1][1]
