from transformers import BertTokenizer

pretrained_model = "https://drive.google.com/u/0/uc?id=1N82kXccrRPvMntddxN2BTzXpYf1BAFHj&export=download&confirm=t" \
                   "&uuid=b3e68e33-b366-40c0-8563-b2b34c83e38a "


def get_pretrained_model():
    return pretrained_model


def init_tokenizer(tokenizer):
    return BertTokenizer.from_pretrained(tokenizer)
