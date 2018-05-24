__author__ = "Benjamin Devillers (bdvllrs)"
__credits__ = ["Benjamin Devillers (bdvllrs)"]
__license__ = "GPL"

import os


class SentimentsSimple:
    def __init__(self, config, sentiment_file):
        self.config = config
        self.sentiment_file = sentiment_file
        self.words = {}
        self.open_file()

    def open_file(self):
        path = os.path.abspath(os.path.join(os.curdir, self.sentiment_file))
        if self.config.debug:
            print("Loading sentiment informations.")
        with open(path, 'r') as file:
            line = file.readline()
            while line:
                if line[0] == 'a':
                    line = line.split('\t')
                    pos_score, neg_score, words = line[2], line[3], line[4].split(' ')
                    for word in words:
                        word = word.split('#')
                        if len(word) > 0:
                            word = word[0]
                            if word in self.words.keys():
                                self.words[word]['pos'] = (self.words[word]['count'] * self.words[word]['pos'] + float(
                                    pos_score)) / (self.words[word]['count'] + 1)
                                self.words[word]['neg'] = (self.words[word]['count'] * self.words[word]['neg'] + float(
                                    neg_score)) / (self.words[word]['count'] + 1)
                                self.words[word]['count'] += 1
                            else:
                                self.words[word] = {
                                    'pos': float(pos_score),
                                    'neg': float(neg_score),
                                    'count': 1
                                }
                line = file.readline()
        if self.config.debug:
            print(len(self.words.keys()), 'words loaded.')

    def pos_score(self, word):
        if word in self.words.keys():
            return self.words[word]['pos']
        return 0

    def neg_score(self, word):
        if word in self.words.keys():
            return self.words[word]['neg']
        return 0

    def sentence_score(self, sentence):
        """
        Get a score from a sentence
        :param sentence: list of words
        """
        pos_score = 0
        neg_score = 0
        for word in sentence:
            pos_score += self.pos_score(word)
            neg_score += self.neg_score(word)
        return pos_score - neg_score
