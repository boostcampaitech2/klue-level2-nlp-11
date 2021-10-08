from importlib import import_module
from tqdm import tqdm
import os
import sys

class MkText():
    def __init__(self):
        sys.path.append("../")
        load_data = getattr(import_module("load_data"), "load_data")
        self.Load_dataset_train = load_data("../../dataset/train/train.csv")
        self.Load_dataset_test = load_data("../../dataset/test/test_data.csv")
        self.sentence_train = self.Load_dataset_train['sentence'].tolist()
        self.sentence_test = self.Load_dataset_test['sentence'].tolist()

    def mkTextfile(self):
        with open('wordlist.txt', 'w') as f:
            for doc in tqdm(self.sentence_train):
                if doc:
                    f.write(doc+ '\n')

            for doc in tqdm(self.sentence_test):
                if doc:
                    f.write(doc + '\n')
            f.close()

if __name__ == '__main__':
    MkText().mkTextfile()