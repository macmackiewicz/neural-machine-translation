import os
import json
import time
import logging

import numpy as np
from cached_property import cached_property
from nltk.translate.bleu_score import corpus_bleu

from nmt.datasets import TextDataset
from nmt.models import Sequence2Sequence


class Sequence2SequenceEvaluator:
    def __init__(self, dataset: TextDataset, train_test_split: float=0.2,
                 **hyperparameters):
        self.dataset = dataset

        # ensure float, since value provided by sagemaker's hyperparameters
        # is serialized as string
        self.train_test_split = float(train_test_split)

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

    @cached_property
    def train_set_length(self):
        return int(len(self.dataset.source) * (1 - self.train_test_split))

    @cached_property
    def x(self):
        return self.dataset.get_sequences('source')

    @cached_property
    def y(self):
        # return self.dataset.encode_output(self.dataset.get_sequences('target'))
        return self.dataset.get_sequences('target')

    @classmethod
    def reconstruct_from_weights(cls, dataset: TextDataset,
                                 model_weights_path: str,
                                 train_test_split: float=0.2):
        # TODO: fix model reconstruction
        dataset.tokenize()
        evaluator = cls(dataset, train_test_split)
        evaluator.model.load_weights(model_weights_path)
        return evaluator

    def train(self, **kwargs):
        y_decode = np.c_[np.zeros(self.y.shape[0]), self.y]
        y_encoded = self.dataset.encode_output(
            np.c_[self.y, np.zeros(self.y.shape[0])]
        )
        self.model.fit([self.x, y_decode], y_encoded,
                       validation_split=self.train_test_split, shuffle=False,
                       **kwargs)

    def save_artifacts(self, output_dir: str):
        timestamp_output_dir = os.path.join(output_dir, self.timestamp)

        config_path = os.path.join(timestamp_output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model.training_model.get_config(), f)

        bleu_score_path = os.path.join(timestamp_output_dir, 'bleu_score.json')
        with open(bleu_score_path, 'w') as f:
            json.dump(self.get_bleu_score(), f)

    def predict_sentence(self, sentence):
        sequence = self.dataset.sentence_to_sequence(sentence)
        predicted_sequence = self.model.predict_sequence(sequence)
        return self.dataset.sequence_to_sentence(predicted_sequence)

    def get_bleu_score(self) -> dict:
        references = []
        predicted_sentences = []

        for sentence in self.dataset.source[self.train_set_length:]:
            references.append(self.dataset.translation_references[sentence])
            predicted_sentences.append(self.predict_sentence(sentence).split())

        bleu_scores = {
            'bleu_1gram': corpus_bleu(references, predicted_sentences,
                                      weights=(1.0, 0, 0, 0)),
            'bleu_2gram': corpus_bleu(references, predicted_sentences,
                                      weights=(0.5, 0.5, 0, 0)),
            'bleu_3gram': corpus_bleu(references, predicted_sentences,
                                      weights=(0.33, 0.33, 0.33, 0)),
            'bleu_4gram': corpus_bleu(references, predicted_sentences,
                                      weights=(0.25, 0.25, 0.25, 0.25)),
        }

        bleu_scores_log = ' '.join(['{}: {};'.format(k, v)
                                    for k, v in bleu_scores.items()])
        self.logger.info(bleu_scores_log)

        return bleu_scores

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
