import csv
import pickle
import numpy as np
from nltk import word_tokenize
import numpy.random as rd
from os import path


class Dataloader:
    def __init__(self, filename, testing_data=False):
        self.file_path = path.abspath(path.join(path.curdir, filename))
        with open(self.file_path, newline='') as file:
            reader = csv.reader(file)
            self.original_lines = [row for row in reader][1:]
        self.lines = list(range(len(self.original_lines)))
        self.word_to_index = {}
        self.index_to_word = []
        self.testing_data = testing_data
        self.preprocess_fn = lambda w, x: x
        self.special_tokens = ['<bos>', '<eos>', '<pad>', '<unk>']
        self.sentiments = None
        self.shuffle_lines()

    def set_sentiments(self, sentiments):
        self.sentiments = sentiments

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
            sentences = self.get(k, no_preprocess=True)[0]
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

    def get(self, index, number=1, random=False, no_preprocess=False, with_sentiments=False):
        """
        Get some lines
        :param index: index of the line
        :param number: number of lines
        :param random: if random
        :param with_sentiments: get simple sentiment analysis
        """
        assert with_sentiments and self.sentiments is not None, "Please use set_sentiments to use sentiment analysis."
        preprocess_fn = lambda x: self.preprocess(x, no_preprocess)
        preprocess_labels_fn = lambda x: self.preprocess_labels(x, no_preprocess)
        if not random:
            batch = list(map(preprocess_fn, self.original_lines[index:index + number]))
            if with_sentiments:
                batch_sentiments = np.array([[i for t in l for i in t] for l in
                                             list(map(self.get_sentiment, self.original_lines[index:index + number]))])
            max_length = self.unify_batch_length(batch)
            if self.testing_data:
                batch_labels = list(map(preprocess_labels_fn, self.original_lines[index:index + number]))
                label_sentences = list(map(lambda x: x[0], batch_labels))
                self.unify_batch_length(label_sentences, max_length)
                label_choice = list(map(lambda x: x[1], batch_labels))
        else:
            indexes = self.lines[index:index + number]
            lines = []
            for i in indexes:
                lines.append(self.original_lines[i])
            batch = list(map(preprocess_fn, lines))
            if with_sentiments:
                batch_sentiments = np.array([[i for t in l for i in t] for l in list(map(self.get_sentiment, lines))])
            max_length = self.unify_batch_length(batch)
            if self.testing_data:
                batch_labels = list(map(preprocess_labels_fn, lines))
                label_sentences = list(map(lambda x: x[0], batch_labels))
                self.unify_batch_length(label_sentences, max_length)
                label_choice = list(map(lambda x: x[1], batch_labels))
        if self.testing_data:
            batch = np.array(batch)
            label_sentences = np.array(label_sentences)
            batch_endings_1 = np.concatenate((batch, np.expand_dims(label_sentences[:, 0, :], axis=1)), axis=1)
            batch_endings_2 = np.concatenate((batch, np.expand_dims(label_sentences[:, 1, :], axis=1)), axis=1)
            if with_sentiments:
                batch_sentiments1 = np.concatenate(
                    (batch_sentiments[:, :8], np.expand_dims(batch_sentiments[:, 8], axis=1)), axis=1)
                batch_sentiments1 = np.concatenate(
                    (batch_sentiments1, np.expand_dims(batch_sentiments[:, 9], axis=1)), axis=1)
                batch_sentiments2 = np.concatenate(
                    (batch_sentiments[:, :8], np.expand_dims(batch_sentiments[:, 10], axis=1)), axis=1)
                batch_sentiments2 = np.concatenate(
                    (batch_sentiments2, np.expand_dims(batch_sentiments[:, 11], axis=1)), axis=1)
            if with_sentiments:
                return batch_endings_1, batch_endings_2, label_choice, batch_sentiments1, batch_sentiments2
            return batch_endings_1, batch_endings_2, label_choice
        if with_sentiments:
            return np.array(batch), batch_sentiments
        return np.array(batch)

    def unify_batch_length(self, batch, max_length=None):
        """
        Set all sequences in batch to the same length
        Get max length of the batch
        """
        if max_length is None:
            max_length = 0
            for story in batch:
                for sentence in story:
                    if len(sentence) > max_length:
                        max_length = len(sentence)
        # Set max length to all sequences
        for k, story in enumerate(batch):
            for i, sentence in enumerate(story):
                batch[k][i] += self.preprocess_fn(self.word_to_index, ['<pad>'] * (max_length - len(sentence)))
        return max_length

    def preprocess_labels(self, line, no_preprocess=False):
        sentences = line[5:7]
        tokenized_sentences = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            if not no_preprocess:
                sentence = self.preprocess_fn(self.word_to_index, sentence)
            tokenized_sentences.append(sentence)
        right_sentence = int(line[7])
        return tokenized_sentences, right_sentence - 1

    def get_sentiment(self, line):
        if self.testing_data:
            sentences = line[1:7]
        else:
            sentences = line[2:]  # remove the 2 first cols id and title
        sentiments = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            sentiments.append(self.sentiments.sentence_score(sentence))
        return sentiments

    def preprocess(self, line, no_preprocess=False):
        if self.testing_data:
            sentences = line[1:5]
        else:
            sentences = line[2:]  # remove the 2 first cols id and title
        tokenized_sentences = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = word_tokenize(sentence)
            if not no_preprocess:
                sentence = self.preprocess_fn(self.word_to_index, sentence)
            tokenized_sentences.append(sentence)
        return tokenized_sentences
