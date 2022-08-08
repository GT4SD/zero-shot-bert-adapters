from typing import List
from transformers import pipeline
from ..utils.utils import init_tokenizer
from .unknown_intents import unknown_intents_set, lemmatization
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import string
import numpy as np
import torch


class ZBERTA:

    def __init__(self, model, tokenizer, dataset=None, z_classes=None, g_classes=None):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.z_classes = z_classes
        self.g_classes = g_classes
        self.tokenizer = tokenizer if type(tokenizer) is not string else init_tokenizer(tokenizer)
        self.model.eval()
        if torch.cuda.is_available():
            self.nlp = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=0)
        else:
            self.nlp = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    def find_new_intents(self, utterances: List = None):
        self.model.eval()
        preds = []
        for i in tqdm(range(0, len(self.dataset if utterances is None else utterances))):
            pred_classes = []
            if utterances is not None:
                classes = unknown_intents_set(utterances)
            else:
                classes = self.z_classes
            for class_set in classes[i]:
                temp = class_set.replace('-', ' ').replace('None', '')
                if temp not in pred_classes and len(temp) > 1:
                    pred_classes.append(temp)
            with torch.no_grad():
                try:
                    pred = self.nlp(self.dataset[i] if utterances is None else utterances[i], pred_classes,
                                    multi_label=False)
                    preds.append(pred['labels'][0])
                except:
                    print("ERROR: no potential intents found")

        return preds

    def compute_semantic_similarity(self, model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"):
        model = SentenceTransformer(model_name)
        preds = self.find_new_intents()
        sim_pred = []
        for i in tqdm(range(0, len(self.dataset))):
            label = self.g_classes[i].replace("_", " ")

            embedding_1 = model.encode(label, convert_to_tensor=True)
            embedding_2 = model.encode(preds[i], convert_to_tensor=True)

            sim_pred.append(util.pytorch_cos_sim(embedding_1, embedding_2).item())

        return sim_pred

    def zero_shot_intents(self):
        sim_preds = self.compute_semantic_similarity()

        count = 0
        for x in sim_preds:
            if x > np.mean(sim_preds) + 0.5 * np.var(sim_preds):
                count += 1

        return count / len(sim_preds)
