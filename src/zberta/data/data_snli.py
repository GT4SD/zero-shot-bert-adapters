from .data_preparation import DataPreparation
import os
from zipfile import ZipFile
import pandas as pd


class DataSNLI(DataPreparation):

    _df_dev = None

    '''
    Initializing SNLI NLI datasets for training data preparation
    '''
    def __init__(self, tokenizer, device, train_eval_split=0.8, batch_size=8):
        super().__init__(tokenizer, device, train_eval_split, batch_size)

        os.system("wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
        file_name = "snli_1.0.zip"
        with ZipFile(file_name, 'r') as zip:
            zip.printdir()
            zip.extractall()

        self.df_train = pd.read_csv('snli_1.0/snli_1.0_train.txt', sep='\t')
        self.df_dev = pd.read_csv('snli_1.0/snli_1.0_dev.txt', sep='\t')
        self.df_test = pd.read_csv('snli_1.0/snli_1.0_test.txt', sep='\t')
        self.df_train = self.df_train[['gold_label','sentence1','sentence2']]
        self.df_dev = self.df_dev[['gold_label','sentence1','sentence2']]
        self.df_test = self.df_test[['gold_label','sentence1','sentence2']]
        self.__class__._df_dev = self.df_dev

    @classmethod
    def _prepare_df(cls, df_train, df_test):

        df_dev = cls._df_dev

        df_train['sentence1'] = df_train['sentence1'].apply(cls.trim_sentence)
        df_train['sentence2'] = df_train['sentence2'].apply(cls.trim_sentence)
        df_dev['sentence1'] = df_dev['sentence1'].apply(cls.trim_sentence)
        df_dev['sentence2'] = df_dev['sentence2'].apply(cls.trim_sentence)
        df_test['sentence1'] = df_test['sentence1'].apply(cls.trim_sentence)
        df_test['sentence2'] = df_test['sentence2'].apply(cls.trim_sentence)
        df_train['sent1'] = '[CLS] ' + df_train['sentence1'] + ' [SEP] '
        df_train['sent2'] = df_train['sentence2'] + ' [SEP]'
        df_dev['sent1'] = '[CLS] ' + df_dev['sentence1'] + ' [SEP] '
        df_dev['sent2'] = df_dev['sentence2'] + ' [SEP]'
        df_test['sent1'] = '[CLS] ' + df_test['sentence1'] + ' [SEP] '
        df_test['sent2'] = df_test['sentence2'] + ' [SEP]'

        df_train = df_train.dropna()
        df_dev = df_dev.dropna()
        df_test = df_test.dropna()

        return df_train, df_dev, df_test

    def iterators(self):
        return super().create_iterator(self.df_train, self.df_test)
