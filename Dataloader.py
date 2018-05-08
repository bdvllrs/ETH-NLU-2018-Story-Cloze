import csv
import pickle
import numpy as np
from nltk import word_tokenize
import numpy.random as rd
from os import path


class Dataloader:
    def __init__(self, filename):
        self.file_path = path.abspath(path.join(path.curdir, filename))
        with open(self.file_path, newline='') as file:
            reader = csv.reader(file)
            self.original_lines = [row for row in reader][1:]
        self.lines = list(range(len(self.original_lines)))
        self.word_to_index = {}
        self.index_to_word = []
        self.preprocess_fn = Dataloader.default_preprocess_line
        self.special_tokens = ['<bos>', '<eos>', '<pad>', '<unk>']
        self.shuffle_lines()

    def set_special_tokens(self, tokens):
        self.special_tokens = tokens

    def set_preprocess_fn(self, preprocess):
        self.preprocess_fn = preprocess

    def shuffle_lines(self):
        rd.shuffle(self.lines)

    def __getitem__(self, item):
        if type(item) == slice:
            return self.get(item.start, item.stop - item.start)
        return self.get(item)

    def __len__(self):
        return len(self.lines)

    def compute_vocab(self):
        """
        Compute the vocab
        """
        words = {}
        for k in range(len(self)):
            sentences = self.get(k, default_preprocess=True)[0]
            # We use all words for vocab even those in the resulting sentence
            for sentence in sentences:
                for word in sentence:
                    if word not in self.special_tokens:
                        if word in words.keys():
                            words[word] += 1
                        else:
                            words[word] = 1
        self.index_to_word = list(list(zip(*sorted(words.items(), key=lambda w: w[1], reverse=True)))[0])
        self.index_to_word = [token for token in self.special_tokens] + self.index_to_word
        for k, word in enumerate(self.index_to_word):
            self.word_to_index[word] = k

    def save_vocab(self, file, size=-1):
        """
        Save vocab into pickle file
        :param file: file location
        :param size: size of the vocab. Default: all vocab
        """
        file_path = path.abspath(path.join(path.curdir, file))
        with open(file_path, 'wb') as file:
            pickle.dump(self.index_to_word[:size], file)

    def load_vocab(self, file, size=-1):
        """
        Load vocab from file
        :param file: location of the vocab file
        :param size: size of the vocab. Default: all vocab
        """
        file_path = path.abspath(path.join(path.curdir, file))
        with open(file_path, 'rb') as file:
            self.index_to_word = pickle.load(file)[:size]
        for k, word in enumerate(self.index_to_word):
            self.word_to_index[word] = k

    def get(self, index, number=1, random=False, default_preprocess=False):
        """
        Get some lines
        :param index: index of the line
        :param number: number of lines
        :param random: if random
        """
        if default_preprocess:
            preprocess_fn = lambda x: Dataloader.default_preprocess_line(self.word_to_index, x)
        else:
            preprocess_fn = lambda x: self.preprocess_fn(self.word_to_index, x)
        if not random:
            batch = list(map(preprocess_fn, self.original_lines[index:index + number]))
        else:
            indexes = self.lines[index:index + number]
            lines = []
            for i in indexes:
                lines.append(self.original_lines[i])
            batch = list(map(preprocess_fn, lines))
        # Set all sequences in batch to the same length
        # Get max length of the batch
        max_length = 0
        for story in batch:
            for sentence in story:
                if len(sentence) > max_length:
                    max_length = len(sentence)
        # Set max length to all sequences
        for k, story in enumerate(batch):
            for i, sentence in enumerate(story):
                batch[k][i] += [self.word_to_index['<pad>']] * (max_length - len(sentence))
        return np.array(batch)

    @staticmethod
    def default_preprocess_line(word_to_index, line):
        sentences = line[2:]  # remove the 2 first cols id and title
        tokenized_sentences = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            tokenized_sentences.append(sentence)
        return tokenized_sentences
