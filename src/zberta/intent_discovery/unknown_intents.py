import spacy_transformers
import spacy
from spacy import displacy
import os
from tqdm import tqdm
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import wordnet

lang = os.environ.get('LANG', 'en')

default_spacy = "en_core_web_trf" if lang == 'en' else "it_core_news_lg"
os.system("python -m spacy download {} --no-deps".format(default_spacy))
try:
    nlp_engine = spacy.load(default_spacy)
except:
    os.system("python3 -m spacy download {} --no-deps".format(default_spacy))
    nlp_engine = spacy.load(default_spacy)
lemmatizer = WordNetLemmatizer()


def en_dependency_parser(dataset):
    classes = []
    for i in tqdm(range(0, len(dataset))):
        action = None
        object = None
        action1 = None
        object1 = None
        action2 = None
        object2 = None
        action3 = None
        object3 = None
        action4 = None
        object4 = None
        doc = nlp_engine(dataset[i].replace('?','').replace('!', '').replace('-', ''))
        deps = displacy.parse_deps(doc)

        for arc in deps['arcs']:
            if arc['label'] == 'dobj':
                start = deps['words'][arc['start']]
                if start['tag'] == 'VERB':
                    action = start['text'].lower()
                end = deps['words'][arc['end']]
                if end['tag'] == 'NOUN':
                    object = end['text'].lower()
            if arc['label'] == 'compound':
                start = deps['words'][arc['start']]
                action1 = start['text'].lower()
                end = deps['words'][arc['end']]
                object1 = end['text'].lower()
            if arc['label'] == 'amod':
                start = deps['words'][arc['start']]
                action2 = start['text'].lower()
                end = deps['words'][arc['end']]
                object2 = end['text'].lower()
            for word in deps['words']:
                if word['tag'] == 'VERB':
                    action3 = word['text'].lower()
                if word['tag'] == 'NOUN':
                    object3 = word['text'].lower()
                if word['tag'] == "ADJ":
                    action4 = word['text'].lower()
                if word['tag'] == "PROPN":
                    object4 = word['text'].lower()

        classes.append([str(action) + "-" + str(object),
                        str(action1) + "-" + str(object1),
                        str(action2) + "-" + str(object2),
                        str(action3) + "-" + str(object3),
                        str(action4) + "-" + str(object4)])
    return classes


def it_dependency_parser(dataset):
    classes = []
    for i in tqdm(range(0, len(dataset['test']))):
        action = None
        object = None
        action1 = None
        object1 = None
        action2 = None
        object2 = None
        action3 = None
        object3 = None
        doc = nlp_engine(dataset['test'][i]['text'].replace('?', '').replace('!', '').replace('-', ''))
        deps = displacy.parse_deps(doc)

        for arc in deps['arcs']:
            if arc['label'] == 'obj':
                start = deps['words'][arc['start']]
                if start['tag'] == 'VERB':
                    action = start['text'].lower()

                end = deps['words'][arc['end']]
                if end['tag'] == 'NOUN':
                    object = end['text'].lower()
            if arc['label'] == 'advmod':
                start = deps['words'][arc['start']]
                action1 = start['text'].lower()
                end = deps['words'][arc['end']]
                object1 = end['text'].lower()
            if arc['label'] == 'aux':
                start = deps['words'][arc['start']]
                action2 = start['text'].lower()
                end = deps['words'][arc['end']]
                object2 = end['text'].lower()
            for word in deps['words']:
                if word['tag'] == 'VERB':
                    action3 = word['text'].lower()
                if word['tag'] == 'NOUN':
                    object3 = word['text'].lower()

        classes.append(
            [str(action) + "-" + str(object), str(action1) + "-" + str(object1), str(action2) + "-" + str(object2),
             str(action3) + "-" + str(object3)])
    return classes


def unknown_intents_set(dataset, lang, spacy_model=None):
    global nlp_engine
    if spacy_model is not None and spacy_model not in default_spacy:
        os.system("python -m spacy download " + spacy_model + " --no-deps")
        nlp_engine = spacy.load(spacy_model)
    if lang == 'en':
        classes = en_dependency_parser(dataset)
    elif lang == 'it':
        classes = it_dependency_parser(dataset)
    else:
        classes = None
    return classes


def _get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatization(intent):
    global lemmatizer
    return " ".join([lemmatizer.lemmatize(w, _get_wordnet_pos(w)) for w in nltk.word_tokenize(intent)])
