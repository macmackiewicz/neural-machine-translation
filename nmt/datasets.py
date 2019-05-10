from typing import Union, Iterable, Sized

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class SourceTargetMixin:
    """
    Allows subscription with 'source' and 'target' keywords
    """
    def __getitem__(self, item):
        if item in ['source', 'target']:
            return getattr(self, item)
        raise TypeError('Subscription is available '
                        'only with "source" and "target" keywords')


class BaseDataset(SourceTargetMixin):
    def __init__(self, source: Union[Iterable, Sized],
                 target: Union[Iterable, Sized],
                 shuffle: bool=True, seed: int=42):
        self.source = source
        self.target = target
        self._validate()
        if shuffle:
            self.shuffle(seed)

    def _validate(self) -> None:
        src_len = len(self.source)
        target_len = len(self.target)
        if src_len != target_len:
            raise TypeError('Number of source rows ({}) does not match '
                            'the number of target rows ({})'.format(src_len,
                                                                    target_len))

    def shuffle(self, seed: int=42) -> None:
        np.random.seed(seed)
        shuffled_indexes = np.random.permutation(len(self.source))
        self.source = self.source[shuffled_indexes]
        self.target = self.target[shuffled_indexes]


class TokenizerPair(SourceTargetMixin):
    def __init__(self, tokenizer_class=Tokenizer):
        self.source = tokenizer_class()
        self.target = tokenizer_class()

    @property
    def is_tokenized(self) -> bool:
        return hasattr(self.source, 'word_index') \
               and hasattr(self.target, 'word_index')


class TextDataset(BaseDataset):
    def __init__(self, source_sentences: Union[Iterable, Sized],
                 target_sentences: Union[Iterable, Sized],
                 shuffle: bool=True):
        super().__init__(source_sentences, target_sentences, shuffle)

        self.tokenizer_pair = TokenizerPair()

    @property
    def source_max_sentence_length(self) -> int:
        return self.max_sentence_length('source')

    @property
    def target_max_sentence_length(self) -> int:
        return self.max_sentence_length('target')

    @property
    def source_vocab_size(self) -> int:
        return self.get_vocab_size('source')

    @property
    def target_vocab_size(self) -> int:
        return self.get_vocab_size('target')

    def get_vocab_size(self, level: str) -> int:
        if not self.tokenizer_pair.is_tokenized:
            raise ValueError('Dataset has not been tokenized yet')
        return len(self.tokenizer_pair[level].word_index) + 1

    def max_sentence_length(self, level: str) -> int:
        return max(len(line.split()) for line in self[level])

    def tokenize(self) -> None:
        if not self.tokenizer_pair.is_tokenized:
            self.tokenizer_pair['source'].fit_on_texts(self.source)
            self.tokenizer_pair['target'].fit_on_texts(self.target)

    def get_sequences(self, level: str) -> np.ndarray:
        if not self.tokenizer_pair.is_tokenized:
            self.tokenize()

        sentences = self.tokenizer_pair[level].texts_to_sequences(self[level])

        return pad_sequences(
            sentences, maxlen=self.max_sentence_length(level), padding='post'
        )

    def encode_output(self, sequences: np.array) -> np.array:
        return to_categorical(sequences, self.target_vocab_size)

    def sequence_to_sentence(self, sequence: Iterable) -> str:
        target_sentence = []
        for idx in sequence:
            word = self.tokenizer_pair['target'].index_word.get(idx)
            if word is None:
                break
            target_sentence.append(word)

        return ' '.join(target_sentence)

    def sentence_to_sequence(self, sentence: str) -> np.ndarray:
        return pad_sequences(
            self.tokenizer_pair['source'].texts_to_sequences([sentence]),
            self.max_sentence_length('source'), padding='post'
        )
