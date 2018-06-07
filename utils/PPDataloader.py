__author__ = "Benjamin Devillers (bdvllrs)"
__credits__ = ["Benjamin Devillers (bdvllrs)"]
__license__ = "GPL"

import os
import json
import numpy as np
import pickle
from nltk import word_tokenize


class PPDataloader:
    """Preprocessed Dataloader"""

    def __init__(self, file):
        """
        :param file: relavite path to a jsonl file
        """
        self.file = os.path.abspath(os.path.join(os.curdir, file))
        self.features = []
        self.original_features = []
        self.output_fn = lambda w, x: x
        self.preprocess_fn = lambda w, x: x
        self.index_to_word = []
        self.word_to_index = {}

        self._get_line_positions()
        self.shuffle_lines()

    def __len__(self):
        return len(self.original_features)

    def set_output_fn(self, output_fn):
        """
        Changes the processing to apply before yielding the data
        :param output_fn:
        """
        self.output_fn = output_fn

    def set_preprocess_fn(self, preprocess_fn):
        """
        Changes the processing to apply before yielding the data
        """
        self.preprocess_fn = preprocess_fn

    def load_vocab(self, file, size=-1, size_percent=None):
        """
        Load vocabulary
        :param file: vocab file
        :param size: size of the vocabulary (default, all vocab)
        :param size_percent: percent of the vocabulary to keep
        :return:
        """
        print('Loading vocab...')
        file_path = os.path.abspath(os.path.join(os.path.curdir, file))
        with open(file_path, 'rb') as file:
            self.index_to_word = pickle.load(file)
        if size_percent is not None:
            size = int(len(self.index_to_word) * size_percent)
        self.index_to_word = self.index_to_word[:size]
        for k, word in enumerate(self.index_to_word):
            self.word_to_index[word] = k
        print('Loaded.')

    def _get_line_positions(self):
        """
        Get seek position of all new lines
        """
        self.file_length = 0
        with open(self.file, 'rb') as file:
            while 1:
                try:
                    features = pickle.load(file)
                except EOFError:
                    break
                self.features.append(features)
        self.features = self.features[:]
        self.original_features = self.features[:]

    def shuffle_lines(self):
        """
        Shuffles the lines
        :return:
        """
        np.random.shuffle(self.features)

    def get(self, item, count=1, random=False):
        """
        Get some values from the dataset
        :param item: index of the value
        :param count: number of items to retrieve
        :param random: if random fetching
        :return: the batch
        """
        batch = []
        k = 0
        while k < count:
            index = (item + k) % len(self.features)
            features = self.features[index] if random else self.original_features[index]
            batch.append(self.preprocess_fn(self.word_to_index, features))
            k += 1
        return self.output_fn(self.word_to_index, batch)

    def get_batch(self, batch_size, n_epochs, random=True):
        """
        Get a generator for batches
        :param batch_size:
        :param n_epochs:
        :param random:
        """
        for epoch in range(n_epochs):
            for k in range(0, len(self), batch_size):
                yield self.get(k, batch_size, random)
            self.shuffle_lines()

