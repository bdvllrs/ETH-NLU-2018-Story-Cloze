import os
import json
import numpy as np


class SNLIDataloader:
    """SNLI Dataloader"""

    def __init__(self, file):
        """
        :param file: relavite path to a jsonl file
        """
        self.file = os.path.abspath(os.path.join(os.curdir, file))
        self.line_positions = []
        self.original_line_positions = []
        self.output_fn = lambda x: x
        self.preprocess_fn = lambda x: x

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

    def _get_line_positions(self):
        """
        Get seek position of all new lines
        """
        self.file_length = 0
        with open(self.file, 'r') as file:
            self.line_positions.append(file.tell())
            line = file.readline()
            while line:
                self.line_positions.append(file.tell())
                line = file.readline()
        self.line_positions = self.line_positions[:-1]
        self.original_line_positions = self.line_positions[:]

    def shuffle_lines(self):
        """
        Shuffles the lines
        :return:
        """
        np.random.shuffle(self.line_positions)

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
            for k in range(count):
                index = (item + k) % len(self)
                position = self.line_positions[index] if random else self.original_line_positions[index]
                file.seek(position)
                line = json.loads(file.readline())
                batch.append(self.preprocess_fn(line))
        return self.output_fn(batch)

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


if __name__ == '__main__':
    dataloader = SNLIDataloader('../data/snli_1.0/snli_1.0_train.jsonl')
    length = len(dataloader)
    print(dataloader.get(0, 3, random=True))
