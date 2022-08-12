import pandas as pd
import click
from tqdm import tqdm
from zs_model_generators import t0_generator, gpt_generator, bloom_generator, keybert_generator

from generate_prompts import get_prompts

def find_model_class(model_name_or_path):

    if "t0" in model_name_or_path.lower():
        return t0_generator
    elif "gpt" in model_name_or_path.lower():
        return gpt_generator
    elif "bloom" in model_name_or_path.lower():
        return bloom_generator
    elif "keybert" in model_name_or_path.lower():
        return keybert_generator
    else:
        raise ValueError(f"{model_name_or_path} is not supported")

@click.command()
@click.option('--url',
              type=str,
              required=True,
              help='Url to input examples.'
)
@click.option('--prompt',
              type=str,
              required=True,
              help='Prompt to be used.'
)
@click.option('--model_name_or_path',
              type=str,
              required=True,
              help='Model to be used for the generation.'
)
@click.option('--cache_dir',
              type=str,
              required=True,
              help='Cache directory for the model.'
)
@click.option('--output_path',
              type=str,
              required=True,
              help='Path to save the generations.'
)
def main(url, prompt, model_name_or_path, cache_dir, output_path):

    model_class = find_model_class(model_name_or_path)
    model = model_class(model_name_or_path, cache_dir)

    cases = get_prompts(url)

    prompts = []
    generations = []

    for case in tqdm(cases):

        input_text = prompt.format(case)
        prompts.append(prompt)
        
        generated = model.generate_text(input_text)

        generations.append(generated[0])

    df = pd.DataFrame.from_dict({"utterance":cases, "prompt":prompts, model_name_or_path:generations})
    df.to_csv(output_path)

if __name__ == "__main__":
    main()
