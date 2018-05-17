import csv
import pickle
import numpy as np
from nltk import word_tokenize
import numpy.random as rd
from os import path


class Data:
    """
    Data object return send to the output_fn callback of `Dataloader`.
    properties:
    - batch: (python list) the returned batch
    - is_testing_data: (bool) if the dataset is a testing set
    - original_lines: whole original dataset
    - preprocessed_lines: whole preprocessed dataset
    - sentiment_lines: sentiment analysis for the batch
    - config: config object
    """

    def __init__(self, batch, is_testing_data, original_lines, preprocessed_lines, sentiment_lines, config):
        self.batch = batch
        self.is_testing_data = is_testing_data
        self.original_lines = original_lines
        self.preprocessed_lines = preprocessed_lines
        self.sentiment_lines = sentiment_lines
        self.config = config

    def __getitem__(self, item):
        return self.get(item)

    def get(self, k):
        return self.batch[k]


class Dataloader:
    """
    Dataloader class.
    ================
    Loads Data for the story cloze test.
    To simply get a `Data` object, use `Dataloader.get(index, amount)`.
    # Callbacks
    ## Pre-processing
    One can control pre-processing by setting the `preprocess_fn` attribute or by calling
    `Dataloader.set_preprocess_fn(callback)`.
    ## Changing the output of get
    The output of the get method is controlled by the `output_fn` attribute, or by calling
    `Dataloader.set_output_fn(callback)`.
    # Vocabs
    To compute the vocab of the Dataset, use `Dataloader.compute_vocab()`.
    Save it with `Dataloader.save_vocab(file, size=-1)` that saves all vocab by default.
    Load it from a file with `Dataloader.load_vocab(file, size=-1)`.
    """

    def __init__(self, config, filename=None, testing_data=False):
        self.config = config
        self.testing_data = testing_data
        if filename is not None:
            self.file_path = path.abspath(path.join(path.curdir, filename))
            with open(self.file_path, newline='') as file:
                reader = csv.reader(file)
                self.original_lines = [row for row in reader][1:]
            self.init_dataset()
            self.tokenize_dataset()

    def set_sentiments(self, sentiments):
        """
        Add a Sentiments instance if want to use sentiment analysis.
        :param sentiments:
        """
        self.sentiments = sentiments
        self.compute_sentiment_dataset()

    def set_special_tokens(self, tokens):
        """
        Sets the token to add to the vocabulary (like <unk>, <pad>...)
        :param tokens:
        """
        self.special_tokens = tokens

    def set_preprocess_fn(self, preprocess):
        """
        Add preprocess function.
        :param preprocess: callback to apply to one sentence. The callback should have the signature
         `preprocess(word_to_index, sentence)` where `word_to_index` is the dict associating words to their tokens and
         `sentence` is a python list of words (strings). The callback should return a list of the preprocessed sentence.
        """
        self.preprocess_fn = preprocess
        self.compute_preprocessed()

    def set_output_fn(self, output_fn):
        """
        Change the function used for output
        :param output_fn: callback, should take a Data object as parameter and return whatever needs to be yielded by get.
        """
        self.output_fn = output_fn

    def shuffle_lines(self):
        """
        Shuffles the batch
        """
        rd.shuffle(self.line_number)

    def __getitem__(self, item):
        if type(item) == slice:
            return self.get(item.start, item.stop - item.start)
        return self.get(item)

    def __len__(self):
        return len(self.line_number)

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

    def save_dataset(self, file):
        """
        Saves the dataset in a pickle file for performance issue
        :param file: relative path to file
        """
        file_path = path.abspath(path.join(path.curdir, file))
        with open(file_path, 'wb') as file:
            pickle.dump(self.original_lines, file)

    def load_dataset(self, file):
        """
        Load a dataset file generated by `Dataloader.save_dataset`.
        :param file: relative path to file
        """
        file_path = path.abspath(path.join(path.curdir, file))
        with open(file_path, 'rb') as file:
            self.original_lines = pickle.load(file)
        self.preprocessed_lines = None
        self.line_number = list(range(len(self.original_lines)))
        self.word_to_index = {}
        self.index_to_word = []
        self.sentiment_lines = []
        self.preprocess_fn = lambda w, x: x
        self.output_fn = lambda data: data
        self.sentiments = None
        self.compute_preprocessed()
        self.shuffle_lines()

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

    def get(self, index, number=1, random=False, no_preprocess=False):
        """
        Get some lines:
        :param index: index in the dataset
        :param number: number of element to return
        :param random: if it needs to be randomized
        :param no_preprocess: if true, does not preprocess the dataset with the defined preprocess_fn.
        :returns: list of shape `number x 5 x sequence length`.
        """
        lines = self.original_lines[:] if no_preprocess else self.preprocessed_lines

        batch = []
        sentiment_batch = []
        for k in range(number):
            i = (index + k) % len(self)
            line_index = i if not random else self.line_number[i]
            batch.append(lines[line_index])
            sentiment_batch.append(self.sentiment_lines[line_index])
        batch = Data(batch, self.testing_data, self.original_lines, self.preprocessed_lines, sentiment_batch,
                     self.config)
        return self.output_fn(batch)

    def get_batch(self, batch_size, epochs, random=True):
        """
        Get a batch
        :param batch_size:
        :param epochs: number of epochs
        :param random: if the sentences should be randomized.
        :return: generator
        """
        for _ in epochs:
            for k in range(0, len(self), batch_size):
                yield self.get(k, batch_size, random)

    def init_dataset(self):
        """
        Internal function. Initialize dataset
        """
        self.preprocessed_lines = None
        self.line_number = list(range(len(self.original_lines)))
        self.word_to_index = {}
        self.index_to_word = []
        self.sentiment_lines = []
        self.preprocess_fn = lambda w, x: x
        self.output_fn = lambda data: data
        self.sentiments = None

        def extract_sentences(x):
            to_return = x[2:7] if not self.testing_data else x[1:9]
            return to_return

        self.original_lines = list(map(extract_sentences, self.original_lines))
        self.compute_preprocessed()
        self.shuffle_lines()

    def tokenize_dataset(self):
        tokenize_fn = lambda x: list(map(lambda s: word_tokenize(s.lower()), x))
        if self.config.debug:
            print('Tokenizing dataset...')
        self.original_lines = list(map(tokenize_fn, self.original_lines))
        if self.config.debug:
            print('Tokenized.')

    def compute_preprocessed(self):
        preprocess_fn = lambda x: list(map(lambda s: self.preprocess_fn(self.word_to_index, s), x))
        self.preprocessed_lines = list(map(preprocess_fn, self.original_lines))

    def compute_sentiment_dataset(self):
        self.sentiment_lines = []
        for batch in range(len(self)):
            scores = []
            for sentence in self.original_lines[batch]:
                score = self.sentiments.sentence_score(sentence)
                scores.append(score)
            self.sentiment_lines.append(scores)
