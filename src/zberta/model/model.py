from transformers import BertConfig
from transformers.adapters import BertModelWithHeads
from .berta import BERTA
import torch
import os
from ..utils.utils import get_pretrained_model


def instantiate_model(labels, out_dim, device, model_name, nli_labels, path=None, dict=False):
    config = BertConfig.from_pretrained(
        model_name,
        num_labels=labels,
    )
    bert_model = BertModelWithHeads.from_pretrained(
        model_name,
        config=config,
    )

    bert_model.to(device)

    bert_model.add_adapter("adapter")

    bert_model.train_adapter(["adapter"])

    bert_model.config.label2id = nli_labels

    HIDDEN_DIM = 512
    OUTPUT_DIM = out_dim

    model = BERTA(bert_model, HIDDEN_DIM, OUTPUT_DIM, dict).to(device)

    if path is None or not os.path.isfile(path):
        os.system("wget -O model.pt '" + get_pretrained_model(out_dim) + "'")
        model.load_state_dict(torch.load('model.pt', map_location=device))
    else:
        model.load_state_dict(torch.load(path, map_location=device))

    return model
