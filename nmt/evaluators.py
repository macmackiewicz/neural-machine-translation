import os
import json
import time
import logging

import numpy as np
from cached_property import cached_property
from keras.utils import Sequence, to_categorical
from nltk.translate.bleu_score import corpus_bleu

from nmt.datasets import TextDataset
from nmt.models import Sequence2Sequence


class SequenceGenerator(Sequence):
    def __init__(self, X, y, target_vocab_size, batch_size=64):
        self.X = X
        self.y = y
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        y_decode = np.c_[np.zeros(batch_y.shape[0]), batch_y]
        y_encoded = to_categorical(
            np.c_[batch_y, np.zeros(batch_y.shape[0])], self.target_vocab_size
        )

        return [batch_x, y_decode], y_encoded


class Sequence2SequenceEvaluator:
    def __init__(self, dataset: TextDataset, train_validation_split: float=0.2,
                 train_test_split: float=0.0, **hyperparameters):
        self.dataset = dataset
        self.train_validation_split = train_validation_split
        self.train_test_split = train_test_split

        self._validate()
        self.timestamp = sagemaker_timestamp()
        # dataset has to be tokenized before its properties can be passed
        # to model constructor
        self.dataset.tokenize()

        self.logger = self.create_logger()

        self.model = Sequence2Sequence.of(dataset.source_vocab_size,
                                          dataset.target_vocab_size,
                                          dataset.target_max_sentence_length,
                                          **hyperparameters
                                          )

    @staticmethod
    def create_logger():
        logger = logging.getLogger(__name__)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

        return logger

    def _validate(self):
        if self.train_validation_split + self.train_test_split > 0.6:
            raise ValueError('Validation and test splits should not exceed 0.6')

    @cached_property
    def train_set_split(self):
        return int(len(self.dataset.source) *
                   (1 - self.train_validation_split - self.train_test_split))

    @cached_property
    def validation_set_split(self):
        return int(len(self.dataset.source) * (1 - self.train_test_split))

    @cached_property
    def x(self):
        return self.dataset.get_sequences('source')

    @cached_property
    def y(self):
        return self.dataset.get_sequences('target')

    @classmethod
    def reconstruct_from_weights(cls, dataset: TextDataset,
                                 model_weights_path: str, **kwargs
                                 ) -> 'Sequence2SequenceEvaluator':
        dataset.tokenize()
        evaluator = cls(dataset, **kwargs)
        evaluator.model.training_model.load_weights(model_weights_path)

        return evaluator

    def train(self, batch_size: int=128, **kwargs) \
            -> 'Sequence2SequenceEvaluator':
        fit_generator = SequenceGenerator(
            self.x[:self.train_set_split],
            self.y[:self.train_set_split],
            self.dataset.target_vocab_size,
            batch_size=batch_size
        )

        validate_generator = SequenceGenerator(
            self.x[self.train_set_split:self.validation_set_split],
            self.y[self.train_set_split:self.validation_set_split],
            self.dataset.target_vocab_size, batch_size=batch_size
        )

        self.model.training_model.fit_generator(
            fit_generator, validation_data=validate_generator, shuffle=True,
            **kwargs
        )
        return self

    def save_artifacts(self, output_dir: str):
        timestamp_output_dir = os.path.join(output_dir, self.timestamp)

        config_path = os.path.join(timestamp_output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model.training_model.get_config(), f)

        self.model.training_model.save_weights(
            '{}/weights.h5'.format(output_dir))

    def predict_sentence(self, sentence):
        sequence = self.dataset.sentence_to_sequence(sentence)
        predicted_sequence = self.model.predict_sequence(sequence)
        return self.dataset.sequence_to_sentence(predicted_sequence)

    def get_bleu_score(self) -> float:
        self.logger.info('Decoding sequences for evaluation with BLEU score')

        references = []
        predicted_sentences = []
        num_predicted_sequences = 0

        for sentence in self.dataset.source[self.validation_set_split:]:
            references.append(self.dataset.translation_references[sentence])
            predicted_sentences.append(self.predict_sentence(sentence).split())
            num_predicted_sequences += 1
            if num_predicted_sequences % 500 == 0:
                self.logger.info(
                    'Decoded {} sequences'.format(num_predicted_sequences))

        bleu_score = corpus_bleu(references, predicted_sentences)

        self.logger.info('BLEU score: {}'.format(bleu_score))

        return bleu_score

####################
# HELPER FUNCTIONS #
####################


def sagemaker_timestamp():
    """
    Return a timestamp with millisecond precision.
    As implemented in sagemaker.utils.sagemaker_timestamp
    """
    moment = time.time()
    moment_ms = repr(moment).split('.')[1][:3]
    return time.strftime("%Y-%m-%d-%H-%M-%S-{}".format(moment_ms), time.gmtime(moment))
