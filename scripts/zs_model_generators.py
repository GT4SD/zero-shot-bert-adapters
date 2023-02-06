from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModel,
)
from keybert import KeyBERT
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class generator:
    def __init__(self, model_name_or_path, cache_dir):

        self.model = AutoModel.from_pretrained(
            model_name_or_path, return_dict_in_generate=True, cache_dir=cache_dir
        ).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )

    def generate_text(
        self,
        text,
        top_k=20,
        top_p=1.0,
        min_length=10,
        max_length=512,
        num_of_sequences=1,
    ):

        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(DEVICE)

        model_output = self.model.generate(
            input_ids,
            do_sample=True,
            num_return_sequences=num_of_sequences,
            no_repeat_ngram_size=2,
            min_length=min_length,
            max_length=max_length,
            top_k=top_k,
            top_p=top_p,
        )

        output = []

        for i in range(num_of_sequences):
            generated_text = model_output["sequences"].cpu()
            generated_text = self.tokenizer.decode(generated_text[i])
            generated_text = generated_text.split("<|endoftext|>")[0]
            generated_text = generated_text.replace("<|pad|>", "")
            generated_text = generated_text.replace("<pad>", "")
            generated_text = generated_text.replace("</s>", "")
            generated_text = generated_text.replace(text, "")
            generated_text = generated_text.strip()

            output.append(generated_text)

        return output


class t0_generator(generator):
    def __init__(self, model_name_or_path, cache_dir):

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, return_dict_in_generate=True, cache_dir=cache_dir
        ).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )


class gpt_generator(generator):
    def __init__(self, model_name_or_path, cache_dir):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, return_dict_in_generate=True, cache_dir=cache_dir
        ).to(DEVICE)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, cache_dir=cache_dir
        )


class keybert_generator:
    def __init__(self, model_name_or_path=None, cache_dir=None):

        self.model = KeyBERT(model_name_or_path, cache_dir=cache_dir)

    def generate_text(self, text):

        keywords = self.model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words=None
        )

        best_match = None
        best_score = 0

        for keyword in keywords:
            if keyword[1] > best_score:
                best_match = keyword[0]
                best_score = keyword[1]
        return [best_match]
