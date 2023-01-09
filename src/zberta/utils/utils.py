from transformers import BertTokenizer

pretrained_model_base_output = "https://drive.google.com/u/0/uc?id=1N82kXccrRPvMntddxN2BTzXpYf1BAFHj&export=download&confirm=t" \
                   "&uuid=b3e68e33-b366-40c0-8563-b2b34c83e38a "

pretrained_model_full_output = "https://drive.google.com/u/0/uc?id=1Bp8OHrNw6TUmxQPApvpwgI8l1j5yXHyl&export=download&confirm=t" \
                     "&uuid=b12ce2db-e0db-4651-88c2-8c1f0e8e5657&at=ACjLJWm3EBiiNI9o7HOMayou8LDr:1673277477283 "


def get_pretrained_model(out_dim):
    if out_dim == 2:
        return pretrained_model_base_output
    return pretrained_model_full_output


def init_tokenizer(tokenizer):
    return BertTokenizer.from_pretrained(tokenizer)
