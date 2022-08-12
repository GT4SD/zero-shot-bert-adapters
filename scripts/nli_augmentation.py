import nltk
from nltk.corpus import wordnet as wn
from keybert import KeyBERT
from tqdm import tqdm
import pandas as pd
import random


def nli_augmentation(df_train):
    nltk.download('wordnet')
    nltk.download('omw')
    kw_model = KeyBERT()
    data = kw_model.extract_keywords(df_train['text'][0], keyphrase_ngram_range=(1, 1), stop_words=None)
    words = []
    word = ""
    for i in tqdm(range(len(df_train))):
        try:
            word = kw_model.extract_keywords(df_train['text'][i], keyphrase_ngram_range=(1, 1), stop_words=None)[0][0]
            wn.synset(wn.synsets(word)[0].name()).definition()
        except:
            words.append(word)

    words_ = []
    word = ""
    out = []
    intro = 'this text is about '

    for i in tqdm(range(len(df_train))):
        try:
            word = kw_model.extract_keywords(df_train['text'][i], keyphrase_ngram_range=(1, 1), stop_words=None)
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
    df_train['hypothesis'] = out
    df_train['gold_label'] = 'entailment'

    df_temp = df_train
    categories = list(df_temp.drop_duplicates("category")['category'])
    dictionary = [ n for n in wn.all_lemma_names() if len(n) > 6]
    for category in tqdm(categories):
        df_category = df_temp[df_train['category'].str.match(category)]
        for i, data in tqdm(df_category.iterrows(), total=df_category.shape[0]):
            rand_word = random.choice(dictionary)
            df_temp = df_temp.append(pd.DataFrame({"text":[data['text']], "category":[category], "hypothesis":[intro + wn.synset(wn.synsets(rand_word)[0].name()).definition()], "gold_label":["contradiction"]}))
    return df_temp
