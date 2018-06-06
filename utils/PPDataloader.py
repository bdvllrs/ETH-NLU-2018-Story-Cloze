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
        self.line_positions = []
        self.original_line_positions = []
        self.output_fn = lambda w, x: x
        self.preprocess_fn = lambda x: x
        self.index_to_word = []
        self.word_to_index = {}

        self._get_line_positions()
        self.shuffle_lines()

    def __len__(self):
        return len(self.line_positions_neg) + len(self.line_positions_pos)

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

    def load_vocab(self, file, size=-1):
        print('Loading vocab...')
        file_path = os.path.abspath(os.path.join(os.path.curdir, file))
        with open(file_path, 'rb') as file:
            self.index_to_word = pickle.load(file)[:size]
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
                line_pos = file.tell()
                try:
                    pickle.load(file)
                except EOFError:
                    break
                self.original_line_positions.append(line_pos)
        self.line_positions = self.line_positions[:]
        self.original_line_positions = self.line_positions[:]

    def shuffle_lines(self):
        """
        Shuffles the lines
        :return:
        """
        np.random.shuffle(self.line_positions)

    def get(self, item, count=1, random=False, only_contradiction=False):
        """
        Get some values from the dataset
        :param item: index of the value
        :param count: number of items to retrieve
        :param random: if random fetching
        :param only_contradiction: if True, only keeps the contradiction pairs
        :return: the batch
        """
        batch = []
        with open(self.file, 'r') as file:
            k, j = 0, 0
            while k < count:
                if np.random.random() > 0.5:
                    index = (item + j) % len(self.line_positions_pos)
                    position = self.line_positions_pos[index] if random else self.original_line_positions_pos[index]
                else:
                    index = (item + j) % len(self.line_positions_neg)
                    position = self.line_positions_neg[index] if random else self.original_line_positions_neg[index]
                file.seek(position)
                line = json.loads(file.readline())
                if not only_contradiction or line['gold_label'] == 'contradiction':
                    batch.append(self.preprocess_fn(line))
                    k += 1
                j += 1
        return self.output_fn(self.word_to_index, batch)

    def get_batch(self, batch_size, n_epochs, random=True, only_contradiction=False):
        """
        Get a generator for batches
        :param batch_size:
        :param n_epochs:
        :param random:
        :param only_contradiction: if True, only keeps contradiction pairs
        """
        for epoch in range(n_epochs):
            for k in range(0, len(self), batch_size):
                yield self.get(k, batch_size, random, only_contradiction)
            self.shuffle_lines()


if __name__ == '__main__':
    dataloader = SNLIDataloader('../data/snli_1.0/snli_1.0_train.jsonl', True)
    # length = len(dataloader)
    # print(dataloader.get(0, 3, random=True))
