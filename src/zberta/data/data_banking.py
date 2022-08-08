from .data_preparation import DataPreparation
import os
import pandas as pd
from nltk.corpus import wordnet as wn
import nltk
from keybert import KeyBERT
from tqdm import tqdm
import random
from datasets import load_dataset


class DataBanking(DataPreparation):

    '''
    Initializing Banking77-OOS intent datasets for training data preparation
    '''
    def __init__(self, tokenizer, device, train_eval_split=0.8, batch_size=8):
        super().__init__(tokenizer, device, train_eval_split, batch_size)

        os.system("git clone https://github.com/jianguoz/Few-Shot-Intent-Detection.git")
        os.system("cp -r Few-Shot-Intent-Detection/Datasets/BANKING77-OOS BANKING77-OOS")

        with open('BANKING77-OOS/train/seq.in', 'r') as seqs, open('BANKING77-OOS/train/label', 'r') as labels:
            seq = seqs.readlines()
            label = labels.readlines()

        with open("BANKING77-OOS/train/train.csv", "w") as data:
            data.write("text,category\n")
            for s,l in zip(seq, label):
                data.write(s.strip().replace(",", "").replace(".", "") + ',' + l.strip() + '\n')
        with open("BANKING77-OOS/train/train.csv", "r") as data:
            out_train = data.readlines()
        
        with open('BANKING77-OOS/test/seq.in', 'r') as seqs, open('BANKING77-OOS/test/label', 'r') as labels:
            seq = seqs.readlines()
            label = labels.readlines()

        with open("BANKING77-OOS/test/test.csv", "w") as data:
            data.write("text,category\n")
            for s,l in zip(seq, label):
                data.write(s.strip().replace(",", "").replace(".", "") + ',' + l.strip() + '\n')
        
        with open("BANKING77-OOS/test/test.csv", "r") as data:
            out_test = data.readlines()

        with open("BANKING77-OOS/test/test.csv", "r") as data:
            out_test = data.readlines()

        df = pd.read_csv("BANKING77-OOS/train/train.csv", index_col=None)
        df = df.dropna()
        df.to_csv("BANKING77-OOS/train/train.csv", index=False)

        df = pd.read_csv("BANKING77-OOS/test/test.csv")
        df = df.dropna()
        df.to_csv("BANKING77-OOS/test/test.csv", index=False)

        self.df_train = pd.read_csv("BANKING77-OOS/train/train.csv", index_col=None)
        self.df_test = pd.read_csv("BANKING77-OOS/test/test.csv")

    @classmethod
    def _prepare_df(cls, df_train, df_test):
        df_train = df_train[:len(df_train) * cls.train_eval_split]
        df_dev = df_train[len(df_train) * cls.train_eval_split:]

        df_train['text'] = df_train['text'].apply(cls.trim_sentence)
        df_train['hypothesis'] = df_train['hypothesis'].apply(cls.trim_sentence)
        df_dev['text'] = df_dev['text'].apply(cls.trim_sentence)
        df_dev['hypothesis'] = df_dev['hypothesis'].apply(cls.trim_sentence)
        df_test['text'] = df_test['text'].apply(cls.trim_sentence)
        df_test['hypothesis'] = df_test['hypothesis'].apply(cls.trim_sentence)

        df_train['sent1'] = '[CLS] ' + df_train['text'] + ' [SEP] '
        df_train['sent2'] = df_train['hypothesis'] + ' [SEP]'
        df_dev['sent1'] = '[CLS] ' + df_dev['text'] + ' [SEP] '
        df_dev['sent2'] = df_dev['hypothesis'] + ' [SEP]'
        df_test['sent1'] = '[CLS] ' + df_test['text'] + ' [SEP] '
        df_test['sent2'] = df_test['hypothesis'] + ' [SEP]'

        df_train = df_train.dropna()
        df_dev = df_dev.dropna()
        df_test = df_test.dropna()

        return df_train, df_dev, df_test

    '''
    Adapting datasets for NLI purposes by adding hypothesis obtained through wordnet + keybert for synset extraction
    '''
    def _convert_to_nli(self):
        nltk.download('wordnet')
        nltk.download('omw')
        kw_model = KeyBERT()
        data = kw_model.extract_keywords(self.df_train['text'][0], keyphrase_ngram_range=(1, 1), stop_words=None)
        words = []
        word = ""
        for i in tqdm(range(len(self.df_train))):
            try:
                word = kw_model.extract_keywords(self.df_train['text'][i], keyphrase_ngram_range=(1, 1), stop_words=None)[0][0]
                wn.synset(wn.synsets(word)[0].name()).definition()
            except:
                words.append(word)

        words_ = []
        word = ""
        out = []
        intro = 'this text is about '

        for i in tqdm(range(len(self.df_train))):
            try:
                word = kw_model.extract_keywords(self.df_train['text'][i], keyphrase_ngram_range=(1, 1), stop_words=None)
                if word[0][0] not in words:
                    out.append(intro + wn.synset(wn.synsets(word[0][0])[0].name()).definition())
                elif len(word[1][0]) > 3:
                    out.append(intro + wn.synset(wn.synsets(word[1][0])[0].name()).definition())
                else:
                    out.append(intro + word[0][0])
            except:
                words_.append(word[1][0])
                out.append(intro + word[0][0])

        out = [x.replace('(', '').replace(')', '') for x in out]
        out = [x.lower() for x in out]
        self.df_train['hypothesis'] = out
        self.df_train['gold_label'] = 'entailment'

        df_temp = self.df_train
        categories = list(df_temp.drop_duplicates("category")['category'])
        dictionary = [ n for n in wn.all_lemma_names() if len(n) > 6]
        for category in tqdm(categories):
            df_category = df_temp[self.df_train['category'].str.match(category)]
            for i, data in tqdm(df_category.iterrows(), total=df_category.shape[0]):
                rand_word = random.choice(dictionary)
                df_temp = df_temp.append(pd.DataFrame({"text":[data['text']], "category":[category], "hypothesis":[intro + wn.synset(wn.synsets(rand_word)[0].name()).definition()], "gold_label":["contradiction"]}))
        self.df_train = df_temp

    def iterators(self):
        return super().create_iterator(self.df_train, self.df_test)

    '''
    Returning dataset for zero-shot evaluation of ID-OOS from Baning77-OOS dataset
    '''
    def z_iterator(self):
        with open('BANKING77-OOS/id-oos/test/seq.in', 'r') as seqs, open('BANKING77-OOS/id-oos/test/label_original', 'r') as labels:
            seq = seqs.readlines()
            label = labels.readlines()

        with open("BANKING77-OOS/id-oos/test/test.csv", "w") as data:
            data.write("text,category\n")
            for s, l in zip(seq, label):
                data.write(s.strip().replace(",", "").replace(".", "") + ',' + l.strip() + '\n')
        df = pd.read_csv("BANKING77-OOS/id-oos/test/test.csv")
        df = df.dropna()
        df.to_csv("BANKING77-OOS/id-oos/test/test.csv", index=False)

        with open("BANKING77-OOS/id-oos/test/test.csv", "r") as data:
            out_test = data.readlines()

        return load_dataset('csv', data_files={'train': 'BANKING77-OOS/id-oos/test/test.csv', 'test': 'BANKING77-OOS/id-oos/test/test.csv'}, encoding = "ISO-8859-1")
