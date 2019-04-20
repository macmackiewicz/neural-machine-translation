from cached_property import cached_property
from nltk.translate.bleu_score import corpus_bleu

from nmt.datasets import TextDataset
from nmt.models import Sequence2Sequence


class Sequence2SequenceEvaluator:
    def __init__(self, dataset: TextDataset, train_test_split: float=0.2):
        self.dataset = dataset

        # ensure float, since value provided by sagemaker's hyperparameters
        # is serialized as string
        self.train_test_split = float(train_test_split)

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

    @property
    def x_train(self):
        return self.dataset.get_sequences('source')[:self.train_set_length]

    @property
    def x_test(self):
        return self.dataset.get_sequences('source')[self.train_set_length:]

    @property
    def y_train(self):
        return self.dataset.encode_output(
            self.dataset.get_sequences('target')[:self.train_set_length]
        )

    @property
    def y_test(self):
        return self.dataset.encode_output(
            self.dataset.get_sequences('target')[self.train_set_length:]
        )

    def train(self, *args, **kwargs): #pylint: disable=W0613
        self.model.fit(self.x_train, self.y_train,
                       validation_data=(self.x_test, self.y_test), **kwargs)

    def predict_sentence(self, sentence):
        sequence = self.dataset.sentence_to_sequence(sentence)
        predicted_sequence = self.model.predict_sequence(sequence)
        return self.dataset.sequence_to_sentence(predicted_sequence)

    def get_bleu_score(self) -> dict:
        original_sentences = []
        predicted_sentences = []

        for sentence in self.dataset.target[self.train_set_length:]:
            original_sentences.append(sentence.split())
            predicted_sentences.append(self.predict_sentence(sentence).split())

        return {
            'bleu1': corpus_bleu(original_sentences, predicted_sentences,
                                 weights=(1.0, 0, 0, 0)),
            'bleu2': corpus_bleu(original_sentences, predicted_sentences,
                                 weights=(0.5, 0.5, 0, 0)),
            'bleu3': corpus_bleu(original_sentences, predicted_sentences,
                                 weights=(0.33, 0.33, 0.33, 0)),
            'bleu4': corpus_bleu(original_sentences, predicted_sentences,
                                 weights=(0.25, 0.25, 0.25, 0.25)),
        }
