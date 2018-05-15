import os
from tqdm import tqdm
from nltk import word_tokenize

from Dataloader import Dataloader


def main():
    dataset_path = os.path.abspath(os.path.join(os.path.curdir, './data/CBTest/data/'))
    export_file = os.path.abspath(os.path.join(os.path.curdir, './data/CBTest/final.txt'))
    for filename in os.listdir(dataset_path):
        if filename[-9:] == 'train.txt':
            with open(os.path.join(dataset_path, filename), 'r') as file:
                with open(export_file, 'a') as file:
                    for line in file:
                        if len(line) > 1 and int(line[:2]) <= 20:
                            tokenized = word_tokenize(line[2:].lower())
                            file.write(' '.join(tokenized) + '\n')


if __name__ == '__main__':
    # Add stories to text file to calculate sentence embeddings
    export_file = os.path.abspath(os.path.join(os.path.curdir, '../sent2vec-cython-wrapper/final.txt'))
    training_set = Dataloader('data/train_stories.csv')
    training_set.set_special_tokens(['<pad>', '<unk>'])
    training_set.load_vocab('./default.voc')
    for k in tqdm(range(len(training_set))):
        sentences = training_set.get(k)[0]
        with open(export_file, 'a') as file:
            for sentence in sentences:
                sentence = ' '.join(list(sentence))
                file.write(sentence + '\n')
