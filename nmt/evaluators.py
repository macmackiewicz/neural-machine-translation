import os
import json

from cached_property import cached_property
from nltk.translate.bleu_score import corpus_bleu

from nmt.datasets import TextDataset
from nmt.models import Sequence2Sequence
from nmt.utils import sagemaker_timestamp


class Sequence2SequenceEvaluator:
    def __init__(self, dataset: TextDataset, train_test_split: float=0.2):
        self.dataset = dataset

        # ensure float, since value provided by sagemaker's hyperparameters
        # is serialized as string
        self.train_test_split = float(train_test_split)

        self.timestamp = sagemaker_timestamp()
        # dataset has to be tokenized before its properties can be passed
        # to model constructor
        self.dataset.tokenize()

        self.model = Sequence2Sequence(dataset.source_vocab_size,
                                       dataset.target_vocab_size,
                                       dataset.source_max_sentence_length,
                                       dataset.target_max_sentence_length
                                       )

    @cached_property
    def train_set_length(self):
        return int(len(self.dataset.source) * (1 - self.train_test_split))

    @cached_property
    def x(self):
        return self.dataset.get_sequences('source')

    @cached_property
    def y(self):
        return self.dataset.encode_output(self.dataset.get_sequences('target'))

    @classmethod
    def reconstruct_from_weights(cls, dataset: TextDataset,
                                 model_weights_path: str,
                                 train_test_split: float=0.2):
        dataset.tokenize()
        evaluator = cls(dataset, train_test_split)
        evaluator.model.load_weights(model_weights_path)
        return evaluator

    def train(self, **kwargs):
        self.model.fit(self.x, self.y, validation_split=self.train_test_split,
                       shuffle=False, **kwargs)

    def save_artifacts(self, output_dir: str):
        timestamp_output_dir = os.path.join(output_dir, self.timestamp)

        config_path = os.path.join(timestamp_output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model.get_config(), f)

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

        return {
            'bleu_1gram': corpus_bleu(references, predicted_sentences,
                                      weights=(1.0, 0, 0, 0)),
            'bleu_2gram': corpus_bleu(references, predicted_sentences,
                                      weights=(0.5, 0.5, 0, 0)),
            'bleu_3gram': corpus_bleu(references, predicted_sentences,
                                      weights=(0.33, 0.33, 0.33, 0)),
            'bleu_4gram': corpus_bleu(references, predicted_sentences,
                                      weights=(0.25, 0.25, 0.25, 0.25)),
        }
