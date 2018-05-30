__author__ = "Benjamin Devillers (bdvllrs)"
__credits__ = ["Benjamin Devillers (bdvllrs)"]
__license__ = "GPL"

import os
import json
import numpy as np
import pickle
from nltk import word_tokenize


class SNLIDataloaderPairs:
    """SNLI Dataloader"""

    def __init__(self, file):
        """
        :param file: relavite path to a jsonl file
        """
        self.file = os.path.abspath(os.path.join(os.curdir, file))
        self.line_positions_pos = []
        self.line_positions_neg = []
        self.line_positions = []
        self.output_fn = lambda w, x: x
        self.preprocess_fn = lambda x: x
        self.index_to_word = []
        self.lines_id = []
        self.word_to_index = {}

        self._get_line_positions()
        self.shuffle_lines()

    def __len__(self):
        return len(self.line_positions)

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
        positions = {}
        with open(self.file, 'r') as file:
            line_pos = file.tell()
            line = file.readline()
            while line:
                json_line = json.loads(line)
                if json_line['gold_label'] != '-':
                    pair_id = json_line['pairID'][:-1]
                    if pair_id not in positions.keys():
                        positions[pair_id] = {'neg': None, 'pos': None}
                    if json_line['gold_label'] == "contradiction":
                        positions[pair_id]['neg'] = line_pos
                    elif json_line['gold_label'] == "neutral":
                        positions[pair_id]['pos'] = line_pos
                line_pos = file.tell()
                line = file.readline()
        for position in positions.values():
            if position['pos'] is not None and position['neg'] is not None:
                self.line_positions.append(position)
        self.lines_id = list(range(len(self.line_positions)))

    def shuffle_lines(self):
        """
        Shuffles the lines
        :return:
        """
        np.random.shuffle(self.lines_id)

    def get(self, item, count=1, random=False):
        """
        Get some values from the dataset
        :param item: index of the value
        :param count: number of items to retrieve
        :param random: if random fetching
        :return: the batch
        """
        batch = []
        with open(self.file, 'r') as file:
            k = 0
            while k < count:
                index = (item + k) % len(self.line_positions)
                positions = self.line_positions[self.lines_id[index]] if random else self.line_positions[index]
                position_pos = positions['pos']
                position_neg = positions['neg']
                file.seek(position_pos)
                line_pos = json.loads(file.readline())
                file.seek(position_neg)
                line_neg = json.loads(file.readline())
                batch.append([self.preprocess_fn(line_pos), self.preprocess_fn(line_neg)])
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

