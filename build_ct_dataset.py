import os
from nltk import word_tokenize


if __name__ == '__main__':
    dataset_path = os.path.abspath(os.path.join(os.path.curdir, './data/CBTest/data/'))
    export_file = os.path.abspath(os.path.join(os.path.curdir, './data/CBTest/final.txt'))
    for filename in os.listdir(dataset_path):
        if filename[-9:] == 'train.txt':
            with open(os.path.join(dataset_path, filename), 'r') as file:
                with open(export_file, 'a') as export_file:
                    for line in file:
                        if len(line) > 1 and int(line[:2]) <= 20:
                            tokenized = word_tokenize(line[2:].lower())
                            export_file.write(' '.join(tokenized) + '\n')
