__author__ = "Benjamin Devillers (bdvllrs)"
__credits__ = ["Benjamin Devillers (bdvllrs)"]
__license__ = "GPL"

import numpy as np
import os
import random


class Sentiments:
    def __init__(self, config, sentiment_folder):
        self.config = config
        self.sentiment_folder = sentiment_folder
        self.train_files = {
            'pos': [],
            'neg': []
        }
        self.test_files = {
            'pos': [],
            'neg': []
        }
        self.vocab_size = config.sentiment_analysis.vocab_size
        self.word_to_index = {}
        self.index_to_word = []
        self.special_tokens = ['<unk>', '<pad>']
        self.open_file()
        self.get_vocab()

    def tokenize_sentence(self, sentence):
        tokenized = []
        for word in sentence:
            if word not in self.index_to_word:
                word = '<unk>'
            tokenized.append(self.word_to_index[word])
        return tokenized

    def __len__(self):
        return len(self.train_files['pos']) + len(self.train_files['neg'])

    def test_length(self):
        return len(self.test_files['pos']) + len(self.test_files['neg'])

    def open_file(self):
        path_pos = os.path.abspath(os.path.join(os.curdir, self.sentiment_folder, './pos/'))
        path_neg = os.path.abspath(os.path.join(os.curdir, self.sentiment_folder, './neg/'))
        if self.config.debug:
            print("Loading sentiment informations.")
        pos_files = os.listdir(path_pos)
        neg_files = os.listdir(path_neg)
        for k, file in enumerate(pos_files):
            if k > 0.2 * len(pos_files):
                self.train_files['pos'].append(os.path.join(path_pos, file))
            else:
                self.test_files['pos'].append(os.path.join(path_pos, file))
        for k, file in enumerate(neg_files):
            if k > 0.2 * len(neg_files):
                self.train_files['neg'].append(os.path.join(path_neg, file))
            else:
                self.test_files['neg'].append(os.path.join(path_neg, file))

    def get_vocab(self, test=False):
        word_dic = {}
        files = self.train_files if not test else self.test_files
        for t in files.keys():
            for file in files[t]:
                with open(file, 'r') as f:
                    lines = f.read().splitlines(keepends=False)
                    for line in lines:
                        words = line.split(' ')
                        for word in words:
                            if word in word_dic.keys():
                                word_dic[word] += 1
                            else:
                                word_dic[word] = 1
        self.index_to_word = list(list(zip(*sorted(word_dic.items(), key=lambda w: w[1], reverse=True)))[0])
        self.index_to_word = [token for token in self.special_tokens] + self.index_to_word
        self.index_to_word = self.index_to_word[:self.vocab_size]
        for k, word in enumerate(self.index_to_word):
            self.word_to_index[word] = k

    def get_random_text(self, test=False):
        sentiment = [1, 0]
        files = self.train_files if not test else self.test_files
        if random.random() > 0.5:
            item = random.randint(0, len(files['pos'])-1)
            file = files['pos'][item]
        else:
            item = random.randint(0, len(files['neg'])-1)
            file = files['neg'][item]
            sentiment = [0, 1]
        with open(file, 'r') as f:
            lines = f.read().splitlines(keepends=False)
            lines = list(map(lambda x: x.split(' '), lines))
            text = []
            for k in range(len(lines)):
                text.extend(self.tokenize_sentence(lines[k]))
        return text, sentiment

    def get_batch(self, batch_size=None, test=False):
        for epoch in range(self.config.n_epochs):
            len_dataset = len(self) if not test else self.test_length()
            for k in range(len_dataset):
                if batch_size is None:
                    batch_size = self.config.batch_size
                batch = []
                batch_label = []
                max_length = 0
                for b in range(batch_size):
                    text, label = self.get_random_text(test=test)
                    if len(text) > max_length:
                        max_length = len(text)
                    batch.append(text)
                    batch_label.append(label)
                for b in range(batch_size):
                    batch[b] += [self.word_to_index['<pad>']] * (max_length - len(batch[b]))
                    batch[b] = batch[b][:self.config.sentiment_analysis.max_length]
                yield np.array(batch), np.array(batch_label)

