from abc import ABCMeta, abstractclassmethod

import numpy as np

from nmt.datasets import TextDataset


class BaseTxtReader(metaclass=ABCMeta):
    def __init__(self, data_path):
        self.data_path = data_path

    def get_txt_file_content(self) -> list:
        with open(self.data_path, mode='r', encoding='utf-8') as file:
            return [line.strip() for line in file.readlines()]

    @abstractclassmethod
    def get_dataset(self):
        raise NotImplementedError


class DelimitedTxtReader(BaseTxtReader):
    def get_dataset(self, delimiter='\t', source_first=False) -> TextDataset:

        content = self.get_txt_file_content()
        pairs = np.array([line.split(delimiter) for line in content])

        source, target = (pairs[:, 0], pairs[:, 1]) if source_first \
            else (pairs[:, 1], pairs[:, 0])

        return TextDataset(source, target)
